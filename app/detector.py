from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import fitz
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    dpi: int = 220
    match_threshold: float = 0.58
    min_scale: float = 0.5
    max_scale: float = 4.0
    dark_threshold: int = 0  # 0 means auto
    nms_iou_threshold: float = 0.2
    max_detection_dim: int = 1700
    min_component_area: int = 3
    max_component_area: int = 140
    max_component_width: int = 12
    min_component_height: int = 3


@dataclass(frozen=True)
class Candidate:
    x: int
    y: int
    w: int
    h: int
    score: float
    angle: float
    scale: float


@dataclass(frozen=True)
class PageResult:
    page_number: int
    count: int
    annotated_path: Path
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class DetectionSummary:
    total_count: int
    page_results: tuple[PageResult, ...]
    output_dir: Path


@dataclass(frozen=True)
class _Component:
    id: int
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float
    cy: float
    hu: tuple[float, ...]


@dataclass(frozen=True)
class _TemplateModel:
    height: float
    left_w: float
    left_h: float
    left_a: float
    left_cy: float
    left_hu: tuple[float, ...]
    right_w: float
    right_h: float
    right_a: float
    right_cy: float
    right_hu: tuple[float, ...]
    dx: float
    dy: float
    aspect_ratio: float


@dataclass(frozen=True)
class _PairCandidate:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float
    left_id: int
    right_id: int


def load_document_pages(path: Path, dpi: int) -> list[np.ndarray]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf_pages(path, dpi=dpi)
    return [_imread_unicode(path)]


def detect_document(
    input_path: Path,
    template_path: Path,
    output_dir: Path,
    config: DetectionConfig | None = None,
) -> DetectionSummary:
    cfg = config or DetectionConfig()
    input_path = input_path.resolve()
    template_path = template_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    template_img = _imread_unicode(template_path)
    template_model = _build_template_model(template_img)

    pages = load_document_pages(input_path, dpi=cfg.dpi)
    page_results: list[PageResult] = []
    total_count = 0

    for page_idx, page_img in enumerate(pages, start=1):
        detect_img, detect_scale = _resize_for_detection(page_img, cfg.max_detection_dim)
        pair_candidates = _detect_on_page(detect_img, template_model, cfg)
        scaled_candidates = _scale_candidates(pair_candidates, 1.0 / detect_scale)

        total_count += len(scaled_candidates)
        annotated = _draw_candidates(page_img, scaled_candidates)

        annotated_path = output_dir / f"{input_path.stem}_page{page_idx:03d}_annotated.png"
        _imwrite_unicode(annotated_path, annotated)

        page_results.append(
            PageResult(
                page_number=page_idx,
                count=len(scaled_candidates),
                annotated_path=annotated_path,
                candidates=tuple(scaled_candidates),
            )
        )

    return DetectionSummary(
        total_count=total_count,
        page_results=tuple(page_results),
        output_dir=output_dir,
    )


def _build_template_model(template_bgr: np.ndarray) -> _TemplateModel:
    binary = _foreground_binary(template_bgr, dark_threshold=0, for_template=True)
    components = _components_from_binary(binary, min_area=2)
    if len(components) < 2:
        raise ValueError("Template must contain the target symbol with 2 visible strokes.")

    components = sorted(components, key=lambda c: c.area, reverse=True)[:2]
    components = sorted(components, key=lambda c: c.cx)
    left, right = components[0], components[1]

    x0 = min(left.x, right.x)
    y0 = min(left.y, right.y)
    x1 = max(left.x + left.w, right.x + right.w)
    y1 = max(left.y + left.h, right.y + right.h)
    height = float(max(1, y1 - y0))
    width = float(max(1, x1 - x0))

    left_cx = (left.cx - x0) / height
    right_cx = (right.cx - x0) / height
    left_cy = (left.cy - y0) / height
    right_cy = (right.cy - y0) / height

    return _TemplateModel(
        height=height,
        left_w=float(left.w) / height,
        left_h=float(left.h) / height,
        left_a=float(left.area) / (height * height),
        left_cy=left_cy,
        left_hu=left.hu,
        right_w=float(right.w) / height,
        right_h=float(right.h) / height,
        right_a=float(right.area) / (height * height),
        right_cy=right_cy,
        right_hu=right.hu,
        dx=right_cx - left_cx,
        dy=abs(right_cy - left_cy),
        aspect_ratio=width / height,
    )


def _detect_on_page(image_bgr: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> list[Candidate]:
    # Baseline binary (dark-contrast) + adaptive color/contrast binary.
    global_binary = _foreground_binary(image_bgr, dark_threshold=cfg.dark_threshold, for_template=False)
    adaptive_binary = _adaptive_color_binary(image_bgr)

    global_components = _filter_page_components(global_binary, model, cfg)
    adaptive_components = _filter_page_components(adaptive_binary, model, cfg)

    global_pairs = _pair_components(global_components, model, cfg) if len(global_components) >= 2 else []
    adaptive_pairs = _pair_components(adaptive_components, model, cfg) if len(adaptive_components) >= 2 else []

    selected: list[_PairCandidate] = []
    if global_pairs:
        selected.extend(_select_best_pairs(global_pairs, cfg.nms_iou_threshold))
    if adaptive_pairs:
        selected.extend(_select_best_pairs(adaptive_pairs, cfg.nms_iou_threshold))

    pair_candidates = _nms_pairs(selected, cfg.nms_iou_threshold)

    result: list[Candidate] = []
    for cand in pair_candidates:
        result.append(
            Candidate(
                x=cand.x,
                y=cand.y,
                w=cand.w,
                h=cand.h,
                score=cand.score,
                angle=0.0,
                scale=cand.scale,
            )
        )
    return result


def _foreground_binary(image_bgr: np.ndarray, dark_threshold: int, for_template: bool) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if for_template:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if np.count_nonzero(binary) < 8:
            binary = np.where(gray < 180, 255, 0).astype(np.uint8)
        if np.count_nonzero(binary) < 8:
            binary = np.where(gray < 230, 255, 0).astype(np.uint8)
        return binary

    threshold = dark_threshold if dark_threshold > 0 else _auto_dark_threshold(gray)
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def _adaptive_color_binary(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    block_size = 31 if min(gray.shape) >= 31 else max(3, (min(gray.shape) // 2) * 2 + 1)
    adaptive_mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        12,
    )

    kernel_size = max(3, int(round(min(gray.shape) * 0.01)))
    kernel_size = min(kernel_size | 1, 15)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    )
    bh_threshold = max(8, int(np.percentile(blackhat, 95)))
    blackhat_mask = np.where(blackhat >= bh_threshold, 255, 0).astype(np.uint8)
    return cv2.bitwise_and(adaptive_mask, blackhat_mask)


def _auto_dark_threshold(gray: np.ndarray) -> int:
    # Conservative default that works well for tiny markup strokes.
    return 90


def _components_from_binary(binary: np.ndarray, min_area: int) -> list[_Component]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[_Component] = []

    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue

        roi = np.where(labels[y : y + h, x : x + w] == idx, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hu = _hu_signature(contour)

        cx, cy = centroids[idx]
        out.append(
            _Component(
                id=idx,
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=int(area),
                cx=float(cx),
                cy=float(cy),
                hu=hu,
            )
        )

    return out


def _hu_signature(contour: np.ndarray) -> tuple[float, ...]:
    hu = cv2.HuMoments(cv2.moments(contour)).flatten()
    sig: list[float] = []
    for value in hu.tolist():
        value = float(value)
        sig.append(float(-np.sign(value) * np.log10(abs(value) + 1e-12)))
    return tuple(sig)


def _hu_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _filter_page_components(binary: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> list[_Component]:
    comps = _components_from_binary(binary, min_area=cfg.min_component_area)
    if not comps:
        return []

    max_area = int(cfg.max_component_area * max(1.0, cfg.max_scale))
    min_height = max(2, int(round(model.height * cfg.min_scale * 0.5)))
    max_height = max(min_height + 1, int(round(model.height * cfg.max_scale * 1.4)))
    max_width = max(cfg.max_component_width, int(round(model.height * cfg.max_scale * 0.8)))

    filtered: list[_Component] = []
    for comp in comps:
        if comp.area > max_area:
            continue
        if comp.h < min_height or comp.h > max_height:
            continue
        if comp.w < 1 or comp.w > max_width:
            continue
        if comp.h < cfg.min_component_height:
            continue
        filtered.append(comp)
    return filtered


def _pair_components(
    components: Sequence[_Component],
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[_PairCandidate]:
    candidates: list[_PairCandidate] = []

    for i, left_raw in enumerate(components):
        for j, right_raw in enumerate(components):
            if i == j:
                continue
            if right_raw.cx <= left_raw.cx:
                continue

            left = left_raw
            right = right_raw

            x0 = min(left.x, right.x)
            y0 = min(left.y, right.y)
            x1 = max(left.x + left.w, right.x + right.w)
            y1 = max(left.y + left.h, right.y + right.h)
            w = int(x1 - x0)
            h = int(y1 - y0)
            if h < 3 or w < 3:
                continue

            scale = float(h) / model.height
            if scale < cfg.min_scale or scale > cfg.max_scale:
                continue

            lf = _normalized_component_features(left, x0=x0, y0=y0, h=h)
            rf = _normalized_component_features(right, x0=x0, y0=y0, h=h)
            pair_dx = rf[3] - lf[3]
            pair_dy = abs(rf[4] - lf[4])
            aspect_ratio = float(w) / float(h)

            if pair_dx < 0.2 or pair_dx > 2.4:
                continue
            if pair_dy > 0.9:
                continue
            if aspect_ratio < 0.35 or aspect_ratio > 4.0:
                continue

            error = _feature_error(
                lf=lf,
                rf=rf,
                pair_dx=pair_dx,
                pair_dy=pair_dy,
                aspect_ratio=aspect_ratio,
                model=model,
            )
            score = 1.0 / (1.0 + error)
            if score < cfg.match_threshold:
                continue

            candidates.append(
                _PairCandidate(
                    x=int(x0),
                    y=int(y0),
                    w=int(w),
                    h=int(h),
                    score=float(score),
                    scale=float(scale),
                    left_id=left.id,
                    right_id=right.id,
                )
            )

    return candidates


def _normalized_component_features(
    comp: _Component,
    x0: int,
    y0: int,
    h: int,
) -> tuple[float, float, float, float, float]:
    hh = float(max(1, h))
    return (
        float(comp.w) / hh,
        float(comp.h) / hh,
        float(comp.area) / (hh * hh),
        float(comp.cx - x0) / hh,
        float(comp.cy - y0) / hh,
    )


def _feature_error(
    lf: tuple[float, float, float, float, float],
    rf: tuple[float, float, float, float, float],
    pair_dx: float,
    pair_dy: float,
    aspect_ratio: float,
    model: _TemplateModel,
) -> float:
    lw, lh, la, _, lcy = lf
    rw, rh, ra, _, rcy = rf
    return (
        abs(lw - model.left_w) * 1.0
        + abs(lh - model.left_h) * 1.0
        + abs(la - model.left_a) * 0.7
        + abs(lcy - model.left_cy) * 0.5
        + abs(rw - model.right_w) * 1.0
        + abs(rh - model.right_h) * 1.0
        + abs(ra - model.right_a) * 0.7
        + abs(rcy - model.right_cy) * 0.5
        + abs(pair_dx - model.dx) * 1.0
        + abs(pair_dy - model.dy) * 0.9
        + abs(aspect_ratio - model.aspect_ratio) * 0.8
    )


def _select_best_pairs(candidates: Sequence[_PairCandidate], iou_threshold: float) -> list[_PairCandidate]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[_PairCandidate] = []
    used_component_ids: set[int] = set()

    for cand in ordered:
        if cand.left_id in used_component_ids or cand.right_id in used_component_ids:
            continue

        overlap = False
        for prev in kept:
            if _pair_iou(cand, prev) >= iou_threshold or _pair_center_too_close(cand, prev):
                overlap = True
                break
        if overlap:
            continue

        kept.append(cand)
        used_component_ids.add(cand.left_id)
        used_component_ids.add(cand.right_id)

    return kept


def _nms_pairs(candidates: Sequence[_PairCandidate], iou_threshold: float) -> list[_PairCandidate]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[_PairCandidate] = []
    for cand in ordered:
        if all(
            _pair_iou(cand, prev) < iou_threshold and not _pair_center_too_close(cand, prev)
            for prev in kept
        ):
            kept.append(cand)
    return kept


def _pair_iou(a: _PairCandidate, b: _PairCandidate) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - inter
    return float(inter) / float(union)


def _pair_center_too_close(a: _PairCandidate, b: _PairCandidate) -> bool:
    ax = a.x + a.w / 2.0
    ay = a.y + a.h / 2.0
    bx = b.x + b.w / 2.0
    by = b.y + b.h / 2.0
    dist = float(np.hypot(ax - bx, ay - by))
    max_dim = float(max(a.w, a.h, b.w, b.h))
    return dist < 1.8 * max_dim


def _scale_candidates(candidates: Sequence[Candidate], factor: float) -> list[Candidate]:
    if abs(factor - 1.0) < 1e-9:
        return list(candidates)

    scaled: list[Candidate] = []
    for cand in candidates:
        scaled.append(
            Candidate(
                x=int(round(cand.x * factor)),
                y=int(round(cand.y * factor)),
                w=int(round(cand.w * factor)),
                h=int(round(cand.h * factor)),
                score=cand.score,
                angle=cand.angle,
                scale=cand.scale,
            )
        )
    return scaled


def _draw_candidates(image: np.ndarray, candidates: Sequence[Candidate]) -> np.ndarray:
    annotated = image.copy()
    for idx, cand in enumerate(candidates, start=1):
        cv2.rectangle(
            annotated,
            (cand.x, cand.y),
            (cand.x + cand.w, cand.y + cand.h),
            (0, 255, 255),
            1,
        )
        cv2.putText(
            annotated,
            str(idx),
            (cand.x, max(10, cand.y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _resize_for_detection(image_bgr: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return image_bgr, 1.0
    scale = float(max_dim) / float(current_max)
    resized = cv2.resize(
        image_bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _load_pdf_pages(path: Path, dpi: int) -> list[np.ndarray]:
    zoom = float(dpi) / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pages: list[np.ndarray] = []
    with fitz.open(path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            buffer = np.frombuffer(pix.samples, dtype=np.uint8)
            rgb = buffer.reshape((pix.height, pix.width, pix.n))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pages.append(bgr)
    return pages


def _imread_unicode(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _imwrite_unicode(path: Path, image: np.ndarray) -> None:
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Cannot encode image as {ext}: {path}")
    encoded.tofile(str(path))
