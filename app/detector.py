from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import cv2
import fitz
import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    dpi: int = 220
    match_threshold: float = 0.50
    min_scale: float = 0.35
    max_scale: float = 8.0
    dark_threshold: int = 0  # 0 means auto
    nms_iou_threshold: float = 0.2
    max_detection_dim: int = 1700
    min_component_area: int = 2
    max_component_area: int = 1200
    max_component_width: int = 48
    min_component_height: int = 2
    max_neighbors: int = 8


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
class _TemplatePart:
    cx: float
    cy: float
    w: float
    h: float
    area_ratio: float


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
    mask: np.ndarray


@dataclass(frozen=True)
class _TemplateModel:
    mask: np.ndarray
    width: int
    height: int
    area: int
    fill_ratio: float
    aspect_ratio: float
    hu: tuple[float, ...]
    component_count: int
    parts: tuple[_TemplatePart, ...]


@dataclass(frozen=True)
class _GroupCandidate:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float
    component_ids: tuple[int, ...]


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
        group_candidates = _detect_on_page(detect_img, template_model, cfg)
        scaled_candidates = _scale_candidates(group_candidates, 1.0 / detect_scale)

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
    if not components:
        raise ValueError("Template must contain visible foreground.")

    max_area = max(comp.area for comp in components)
    major_components = [comp for comp in components if comp.area >= max(2, int(round(max_area * 0.12)))]
    if not major_components:
        major_components = [max(components, key=lambda comp: comp.area)]
    major_components = sorted(major_components, key=lambda comp: comp.area, reverse=True)[:3]
    major_components = sorted(major_components, key=lambda comp: comp.cx)

    x0, y0, x1, y1 = _group_bbox(major_components)
    union_mask = _render_group_mask(major_components, x0, y0, x1 - x0, y1 - y0)
    union_area = int(np.count_nonzero(union_mask))
    if union_area == 0:
        raise ValueError("Template must contain visible foreground.")

    width = int(x1 - x0)
    height = int(y1 - y0)
    parts = tuple(_normalize_part(comp, x0=x0, y0=y0, width=width, height=height, union_area=union_area) for comp in major_components)

    return _TemplateModel(
        mask=union_mask,
        width=width,
        height=height,
        area=union_area,
        fill_ratio=float(union_area) / float(max(1, width * height)),
        aspect_ratio=float(width) / float(max(1, height)),
        hu=_hu_signature(union_mask),
        component_count=len(major_components),
        parts=parts,
    )


def _detect_on_page(image_bgr: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> list[Candidate]:
    binaries = [_foreground_binary(image_bgr, dark_threshold=cfg.dark_threshold, for_template=False)]
    if model.component_count == 1:
        binaries.append(_background_distance_binary(image_bgr))
        binaries.append(_adaptive_color_binary(image_bgr))
    elif model.area >= 80:
        binaries.append(_adaptive_color_binary(image_bgr))

    all_candidates: list[_GroupCandidate] = []
    for binary in binaries:
        components = _filter_page_components(binary, model, cfg)
        if not components:
            continue
        group_candidates = _score_candidate_groups(components, model, cfg)
        if group_candidates:
            all_candidates.extend(group_candidates)

    final_candidates = _nms_groups(all_candidates, cfg.nms_iou_threshold)
    return [
        Candidate(
            x=cand.x,
            y=cand.y,
            w=cand.w,
            h=cand.h,
            score=cand.score,
            angle=0.0,
            scale=cand.scale,
        )
        for cand in final_candidates
    ]


def _foreground_binary(image_bgr: np.ndarray, dark_threshold: int, for_template: bool) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    if for_template:
        bg_color = _estimate_border_background(image_bgr)
        color_dist = np.linalg.norm(image_bgr.astype(np.float32) - bg_color, axis=2)
        color_mask = np.where(color_dist >= max(12.0, float(np.percentile(color_dist, 80))), 255, 0).astype(np.uint8)
        _, gray_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = cv2.bitwise_or(color_mask, gray_mask)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
        if np.count_nonzero(binary) < 8:
            binary = np.where(gray < 235, 255, 0).astype(np.uint8)
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

    combined = cv2.bitwise_and(adaptive_mask, blackhat_mask)
    return cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _background_distance_binary(image_bgr: np.ndarray) -> np.ndarray:
    bg_color = _estimate_border_background(image_bgr)
    color_dist = np.linalg.norm(image_bgr.astype(np.float32) - bg_color, axis=2)
    threshold = max(10.0, float(np.percentile(color_dist, 85)))
    binary = np.where(color_dist >= threshold, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _estimate_border_background(image_bgr: np.ndarray) -> np.ndarray:
    top = image_bgr[0, :, :]
    bottom = image_bgr[-1, :, :]
    left = image_bgr[:, 0, :]
    right = image_bgr[:, -1, :]
    border = np.concatenate((top, bottom, left, right), axis=0)
    return np.median(border.astype(np.float32), axis=0)


def _auto_dark_threshold(gray: np.ndarray) -> int:
    percentile = float(np.percentile(gray, 0.35))
    return int(np.clip(percentile + 18.0, 70.0, 140.0))


def _components_from_binary(binary: np.ndarray, min_area: int) -> list[_Component]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[_Component] = []

    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < min_area:
            continue

        mask = np.where(labels[y : y + h, x : x + w] == idx, 255, 0).astype(np.uint8)
        cx, cy = centroids[idx]
        components.append(
            _Component(
                id=idx,
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=int(area),
                cx=float(cx),
                cy=float(cy),
                mask=mask,
            )
        )

    return components


def _filter_page_components(binary: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> list[_Component]:
    components = _components_from_binary(binary, min_area=cfg.min_component_area)
    if not components:
        return []

    min_area = max(cfg.min_component_area, int(round(model.area * (cfg.min_scale**2) * 0.03)))
    max_area = max(cfg.max_component_area, int(round(model.area * (cfg.max_scale**2) * 1.4)))
    min_height = max(cfg.min_component_height, int(round(model.height * cfg.min_scale * 0.18)))
    max_height = max(min_height + 1, int(round(model.height * cfg.max_scale * 1.2)))
    max_width = max(cfg.max_component_width, int(round(model.width * cfg.max_scale * 1.2)))

    filtered: list[_Component] = []
    for comp in components:
        if comp.area < min_area or comp.area > max_area:
            continue
        if comp.h < min_height or comp.h > max_height:
            continue
        if comp.w < 1 or comp.w > max_width:
            continue
        filtered.append(comp)
    return filtered


def _score_candidate_groups(
    components: Sequence[_Component],
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[_GroupCandidate]:
    candidates: list[_GroupCandidate] = []
    seen_group_ids: set[tuple[int, ...]] = set()

    for group in _generate_candidate_groups(components, model, cfg):
        group_ids = tuple(sorted(comp.id for comp in group))
        if group_ids in seen_group_ids:
            continue
        seen_group_ids.add(group_ids)

        candidate = _score_group(group, model, cfg)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def _generate_candidate_groups(
    components: Sequence[_Component],
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[tuple[_Component, ...]]:
    components = sorted(components, key=lambda comp: comp.cx)
    groups: list[tuple[_Component, ...]] = []

    if model.component_count == 1:
        groups.extend((comp,) for comp in components)

    max_span = max(model.width, model.height) * cfg.max_scale * 2.4
    for idx, anchor in enumerate(components):
        neighbors: list[_Component] = []
        for other in components[idx + 1 :]:
            if other.x - anchor.x > max_span:
                break
            if abs(other.cy - anchor.cy) > max(anchor.h, other.h, model.height * cfg.max_scale):
                continue
            neighbors.append(other)
            if len(neighbors) >= cfg.max_neighbors:
                break

        if model.component_count <= 2:
            for neighbor in neighbors:
                groups.append((anchor, neighbor))
        elif model.component_count >= 3 and len(neighbors) >= 2:
            for pair in combinations(neighbors, 2):
                groups.append((anchor, pair[0], pair[1]))

    return groups


def _score_group(
    group: Sequence[_Component],
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> _GroupCandidate | None:
    x0, y0, x1, y1 = _group_bbox(group)
    width = int(x1 - x0)
    height = int(y1 - y0)
    if width < 2 or height < 2:
        return None

    scale_h = float(height) / float(max(1, model.height))
    scale_w = float(width) / float(max(1, model.width))
    scale = 0.5 * (scale_h + scale_w)
    if scale < cfg.min_scale or scale > cfg.max_scale:
        return None

    union_mask = _render_group_mask(group, x0, y0, width, height)
    union_area = int(np.count_nonzero(union_mask))
    if union_area == 0:
        return None

    resized_mask = cv2.resize(union_mask, (model.width, model.height), interpolation=cv2.INTER_NEAREST)
    resized_mask = np.where(resized_mask > 0, 255, 0).astype(np.uint8)

    iou = _mask_iou(resized_mask, model.mask)
    dilated_iou = _mask_iou(
        cv2.dilate(resized_mask, np.ones((3, 3), dtype=np.uint8)),
        cv2.dilate(model.mask, np.ones((3, 3), dtype=np.uint8)),
    )
    hu_sim = _similarity_from_distance(_hu_distance(_hu_signature(resized_mask), model.hu), scale=2.4)
    aspect_sim = _ratio_similarity(float(width) / float(height), model.aspect_ratio)
    fill_ratio = float(union_area) / float(max(1, width * height))
    fill_sim = max(0.0, 1.0 - abs(fill_ratio - model.fill_ratio) / 0.55)
    layout_sim = _layout_similarity(group, x0, y0, width, height, union_area, model)

    score = (
        0.34 * iou
        + 0.18 * dilated_iou
        + 0.18 * hu_sim
        + 0.12 * aspect_sim
        + 0.08 * fill_sim
        + 0.10 * layout_sim
    )
    if score < cfg.match_threshold:
        return None

    return _GroupCandidate(
        x=x0,
        y=y0,
        w=width,
        h=height,
        score=float(score),
        scale=float(scale),
        component_ids=tuple(sorted(comp.id for comp in group)),
    )


def _layout_similarity(
    group: Sequence[_Component],
    x0: int,
    y0: int,
    width: int,
    height: int,
    union_area: int,
    model: _TemplateModel,
) -> float:
    if not model.parts:
        return 1.0

    if len(group) != len(model.parts):
        if model.component_count == 1 and len(group) == 2:
            # Connected templates can be split by rasterization in low-quality pages.
            return 0.72
        return 0.0

    norm_parts = [
        _normalize_part(comp, x0=x0, y0=y0, width=width, height=height, union_area=union_area)
        for comp in sorted(group, key=lambda comp: comp.cx)
    ]

    error = 0.0
    for candidate_part, template_part in zip(norm_parts, model.parts):
        error += abs(candidate_part.cx - template_part.cx) * 1.2
        error += abs(candidate_part.cy - template_part.cy) * 0.9
        error += abs(candidate_part.w - template_part.w) * 1.0
        error += abs(candidate_part.h - template_part.h) * 1.0
        error += abs(candidate_part.area_ratio - template_part.area_ratio) * 1.0
    return _similarity_from_distance(error, scale=2.5)


def _normalize_part(
    comp: _Component,
    x0: int,
    y0: int,
    width: int,
    height: int,
    union_area: int,
) -> _TemplatePart:
    ww = float(max(1, width))
    hh = float(max(1, height))
    return _TemplatePart(
        cx=float(comp.cx - x0) / ww,
        cy=float(comp.cy - y0) / hh,
        w=float(comp.w) / ww,
        h=float(comp.h) / hh,
        area_ratio=float(comp.area) / float(max(1, union_area)),
    )


def _similarity_from_distance(distance: float, scale: float) -> float:
    return 1.0 / (1.0 + max(0.0, distance) * scale)


def _ratio_similarity(value: float, reference: float) -> float:
    value = max(value, 1e-6)
    reference = max(reference, 1e-6)
    return float(np.exp(-abs(np.log(value / reference))))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    inter = int(np.count_nonzero(a & b))
    return float(inter) / float(union)


def _group_bbox(group: Sequence[_Component]) -> tuple[int, int, int, int]:
    x0 = min(comp.x for comp in group)
    y0 = min(comp.y for comp in group)
    x1 = max(comp.x + comp.w for comp in group)
    y1 = max(comp.y + comp.h for comp in group)
    return int(x0), int(y0), int(x1), int(y1)


def _render_group_mask(
    group: Sequence[_Component],
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=np.uint8)
    for comp in group:
        rx = comp.x - x0
        ry = comp.y - y0
        roi = canvas[ry : ry + comp.h, rx : rx + comp.w]
        np.maximum(roi, comp.mask, out=roi)
    return canvas


def _hu_signature(mask: np.ndarray) -> tuple[float, ...]:
    moments = cv2.moments(mask, binaryImage=True)
    hu = cv2.HuMoments(moments).flatten()
    signature: list[float] = []
    for value in hu.tolist():
        value = float(value)
        signature.append(float(-np.sign(value) * np.log10(abs(value) + 1e-12)))
    return tuple(signature)


def _hu_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _nms_groups(candidates: Sequence[_GroupCandidate], iou_threshold: float) -> list[_GroupCandidate]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda cand: cand.score, reverse=True)
    kept: list[_GroupCandidate] = []
    used_component_sets: set[tuple[int, ...]] = set()

    for cand in ordered:
        if cand.component_ids in used_component_sets:
            continue

        if any(_group_iou(cand, prev) >= iou_threshold or _group_center_too_close(cand, prev) for prev in kept):
            continue

        kept.append(cand)
        used_component_sets.add(cand.component_ids)

    return kept


def _group_iou(a: _GroupCandidate, b: _GroupCandidate) -> float:
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


def _group_center_too_close(a: _GroupCandidate, b: _GroupCandidate) -> bool:
    ax = a.x + a.w / 2.0
    ay = a.y + a.h / 2.0
    bx = b.x + b.w / 2.0
    by = b.y + b.h / 2.0
    dist = float(np.hypot(ax - bx, ay - by))
    min_dim = float(min(a.w, a.h, b.w, b.h))
    return dist < 0.9 * min_dim


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
