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
    match_threshold: float = 0.45
    min_scale: float = 0.18
    max_scale: float = 1.0
    dark_threshold: int = 0  # 0 means auto
    nms_iou_threshold: float = 0.35
    num_scales: int = 12  # number of scale steps for matchTemplate sweep
    max_detection_dim: int = 2500
    min_component_area: int = 2
    max_component_area: int = 1800
    max_component_width: int = 72
    min_component_height: int = 2
    max_neighbors: int = 10
    scope: tuple[tuple[float, float], ...] | None = None
    scope_min_overlap: float = 0.6
    color_sensitivity: str = "auto"
    uniform_size_assist: bool = False
    chromatic_gate_on: float = 0.45
    hue_gate_max_diff: float = 28.0
    class_assignment_margin: float = 0.05
    global_non_overlap: bool = True
    exclude_unclassified: bool = True
    debug_artifacts: bool = False


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
    class_totals: tuple[tuple[str, int], ...] = ()
    unclassified_count: int = 0


@dataclass(frozen=True)
class MarkupClass:
    name: str
    template_path: Path


@dataclass(frozen=True)
class _AssignedCandidate:
    class_index: int
    candidate: Candidate
    margin: float
    score: float
    size_consistency: float


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
    aspect_ratio: float
    hu: tuple[float, ...]
    component_count: int
    parts: tuple[_TemplatePart, ...]
    color_profile: "_ColorProfile"
    color_weight: float
    ignore_high_saturation: bool
    circularity: float
    solidity: float


@dataclass(frozen=True)
class _GroupCandidate:
    x: int
    y: int
    w: int
    h: int
    score: float
    scale: float
    component_ids: tuple[int, ...]


@dataclass(frozen=True)
class _ColorProfile:
    fg_lab_mean: tuple[float, float, float]
    fg_lab_std: tuple[float, float, float]
    fg_hue_mean: float
    fg_hue_spread: float
    fg_sat_mean: float
    fg_sat_std: float
    fg_chroma_mean: float
    fg_chroma_std: float
    chromatic: bool


@dataclass(frozen=True)
class _ColorPolicy:
    weight: float
    min_similarity: float
    binary_threshold: float
    tolerance_scale: float
    use_color_map: bool


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

    return detect_document_multi(
        input_path=input_path,
        markups=(MarkupClass(name=template_path.stem or "markup", template_path=template_path),),
        output_dir=output_dir,
        config=cfg,
    )


def detect_document_multi(
    input_path: Path,
    markups: Sequence[MarkupClass],
    output_dir: Path,
    config: DetectionConfig | None = None,
) -> DetectionSummary:
    cfg = config or DetectionConfig()
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not markups:
        raise ValueError("At least one markup template is required.")

    models: list[_TemplateModel] = []
    class_names: list[str] = []
    template_images: list[np.ndarray] = []
    for markup in markups:
        template_path = markup.template_path.resolve()
        template_img = _imread_unicode(template_path)
        template_images.append(template_img)
        models.append(_build_template_model(template_img))
        class_names.append(markup.name.strip() or template_path.stem or "markup")

    debug_dir = output_dir / "_debug"
    if cfg.debug_artifacts:
        debug_dir.mkdir(parents=True, exist_ok=True)
        for class_idx, (class_name, model, template_img) in enumerate(zip(class_names, models, template_images), start=1):
            _export_template_debug(debug_dir, class_idx, class_name, template_img, model)

    pages = load_document_pages(input_path, dpi=cfg.dpi)
    page_results: list[PageResult] = []
    class_totals = [0 for _ in models]
    unclassified_total = 0

    for page_idx, page_img in enumerate(pages, start=1):
        detect_img, detect_scale = _resize_for_detection(page_img, cfg.max_detection_dim)
        scope_polygon = _resolve_scope_polygon(detect_img.shape[:2], cfg.scope)
        scope_rect = _polygon_bbox(scope_polygon)

        crop_x = 0
        crop_y = 0
        detect_view = detect_img
        if scope_rect is not None:
            crop_x, crop_y, crop_x1, crop_y1 = scope_rect
            detect_view = detect_img[crop_y:crop_y1, crop_x:crop_x1]

        per_class_candidates: list[list[Candidate]] = []
        for model in models:
            raw = _detect_on_page(detect_view, model, cfg)
            offset = _offset_candidates(raw, crop_x, crop_y)
            scoped = _filter_candidates_by_scope(offset, scope_polygon, cfg.scope_min_overlap)
            scaled = _scale_candidates(scoped, 1.0 / detect_scale)
            per_class_candidates.append(scaled)

        if len(models) == 1:
            merged_candidates = per_class_candidates[0]
            if cfg.global_non_overlap:
                merged_candidates = _dedupe_candidates(merged_candidates, min(cfg.nms_iou_threshold, 0.10))
            class_counts = [len(merged_candidates)]
            unclassified = 0
        else:
            merged_candidates, class_counts, unclassified = _merge_multi_class_candidates(
                page_img,
                models,
                per_class_candidates,
                cfg,
            )
        for idx, count in enumerate(class_counts):
            class_totals[idx] += count
        unclassified_total += unclassified

        page_scope = None
        if scope_polygon is not None:
            page_scope = tuple((int(round(x / detect_scale)), int(round(y / detect_scale))) for x, y in scope_polygon)

        annotated = _draw_candidates(page_img, merged_candidates, page_scope)
        annotated_path = output_dir / f"{input_path.stem}_page{page_idx:03d}_annotated.png"
        _imwrite_unicode(annotated_path, annotated)

        if cfg.debug_artifacts:
            for class_idx, (class_name, model) in enumerate(zip(class_names, models), start=1):
                _export_page_debug(
                    debug_dir=debug_dir,
                    page_idx=page_idx,
                    class_idx=class_idx,
                    class_name=class_name,
                    detect_view=detect_view,
                    detect_scale=detect_scale,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    page_img=page_img,
                    model=model,
                    cfg=cfg,
                    final_candidates=per_class_candidates[class_idx - 1],
                )
        page_results.append(
            PageResult(
                page_number=page_idx,
                count=len(merged_candidates),
                annotated_path=annotated_path,
                candidates=tuple(merged_candidates),
            )
        )

    return DetectionSummary(
        total_count=sum(page.count for page in page_results),
        page_results=tuple(page_results),
        output_dir=output_dir,
        class_totals=tuple((name, count) for name, count in zip(class_names, class_totals)),
        unclassified_count=unclassified_total,
    )


def _build_template_model(template_bgr: np.ndarray) -> _TemplateModel:
    prep_template, ignore_high_saturation = _prepare_template_image(template_bgr)
    binary = _foreground_binary(prep_template, dark_threshold=0, for_template=True)
    components = _components_from_binary(binary, min_area=2)
    if not components:
        raise ValueError("Template must contain visible foreground.")

    major_components = _select_template_components(components, prep_template.shape[:2])

    x0, y0, x1, y1 = _group_bbox(major_components)
    union_mask = _render_group_mask(major_components, x0, y0, x1 - x0, y1 - y0)
    union_area = int(np.count_nonzero(union_mask))
    if union_area == 0:
        raise ValueError("Template must contain visible foreground.")

    width = int(x1 - x0)
    height = int(y1 - y0)
    template_crop = prep_template[y0:y1, x0:x1]
    parts = tuple(
        _normalize_part(comp, x0=x0, y0=y0, width=width, height=height, union_area=union_area)
        for comp in major_components
    )
    color_profile, color_weight = _template_color_signature(template_crop, union_mask)
    circularity = _mask_circularity(union_mask)
    solidity = _mask_solidity(union_mask)

    return _TemplateModel(
        mask=union_mask,
        width=width,
        height=height,
        area=union_area,
        aspect_ratio=float(width) / float(max(1, height)),
        hu=_hu_signature(union_mask),
        component_count=len(major_components),
        parts=parts,
        color_profile=color_profile,
        color_weight=color_weight,
        ignore_high_saturation=ignore_high_saturation,
        circularity=circularity,
        solidity=solidity,
    )


def _select_template_components(
    components: Sequence[_Component],
    image_shape: tuple[int, int],
) -> list[_Component]:
    if not components:
        return []

    total_area = float(sum(comp.area for comp in components))
    height, width = image_shape
    cx_mid = width / 2.0
    cy_mid = height / 2.0
    max_center_dist = max(1.0, float(np.hypot(cx_mid, cy_mid)))

    scored: list[tuple[float, _Component]] = []
    for comp in components:
        center_dist = float(np.hypot(comp.cx - cx_mid, comp.cy - cy_mid)) / max_center_dist
        center_score = max(0.0, 1.0 - center_dist)
        circ = _mask_circularity(comp.mask)
        solid = _mask_solidity(comp.mask)
        area_ratio = comp.area / max(1.0, total_area)
        blob_score = 0.44 * area_ratio + 0.28 * center_score + 0.18 * circ + 0.10 * solid
        scored.append((blob_score, comp))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_comp = scored[0]
    best_area_ratio = best_comp.area / max(1.0, total_area)
    best_circularity = _mask_circularity(best_comp.mask)
    best_solidity = _mask_solidity(best_comp.mask)
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    # If one centered, dominant, filled blob stands out, treat the template as that blob
    # even when the crop also contains nearby labels or small fragments.
    if (
        best_area_ratio >= 0.42
        and best_circularity >= 0.48
        and best_solidity >= 0.62
        and best_score >= second_score + 0.08
    ):
        return [best_comp]

    max_area = max(comp.area for comp in components)
    major_components = [comp for comp in components if comp.area >= max(2, int(round(max_area * 0.12)))]
    if not major_components:
        major_components = [max(components, key=lambda comp: comp.area)]
    major_components = sorted(major_components, key=lambda comp: comp.area, reverse=True)[:3]
    return sorted(major_components, key=lambda comp: comp.cx)


def _detect_on_page(image_bgr: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> list[Candidate]:
    working_bgr = image_bgr
    color_policy = _resolve_color_policy(model, cfg)
    all_candidates: list[_GroupCandidate] = []
    if _is_simple_blob_model(model):
        # Human counting of filled marks is mostly color + blob geometry + size consistency.
        # Generic component grouping and template correlation overfire on text and map details,
        # so simple filled markups use a blob-first detection path.
        all_candidates.extend(_simple_blob_candidates(working_bgr, model, cfg))
        all_candidates.extend(_dense_blob_candidates(working_bgr, model, cfg))
        all_candidates.extend(_centered_blob_proposals(working_bgr, model, cfg))
    else:
        binaries = [
            _foreground_binary(working_bgr, dark_threshold=cfg.dark_threshold, for_template=False),
            _background_distance_binary(working_bgr),
            _adaptive_color_binary(working_bgr),
            _clahe_binary(working_bgr),
        ]
        if color_policy.use_color_map:
            binaries.append(_template_color_binary(working_bgr, model, cfg))

        for binary in binaries:
            components = _filter_page_components(binary, model, cfg)
            if not components:
                continue
            group_candidates = _score_candidate_groups(components, working_bgr, model, cfg)
            if group_candidates:
                all_candidates.extend(group_candidates)

        # --- pixel-level template matching passes (primary signal) ---
        tmpl_mask = model.mask  # already a 2-D uint8 binary

        match_candidates = _template_match_candidates(working_bgr, tmpl_mask, model, cfg)
        all_candidates.extend(match_candidates)

        edge_candidates = _edge_match_candidates(working_bgr, tmpl_mask, model, cfg)
        all_candidates.extend(edge_candidates)

    final_candidates = _nms_groups(all_candidates, cfg.nms_iou_threshold)
    
    # --- Post-NMS Color Penalty ---
    # We apply color checking here because cv2.kmeans inside the matching loops 
    # would run thousands of times. Here it only runs on the NMS peaks.
    filtered_results: list[Candidate] = []
    for cand in final_candidates:
        filtered, _ = _post_filter_candidate(cand, image_bgr, model, cfg)
        if filtered is not None:
            filtered_results.append(filtered)

    if cfg.uniform_size_assist and filtered_results:
        if _is_simple_blob_model(model):
            # For filled dots, box normalization can inflate candidates into large overlapping circles
            # and collapse many true hits into a small count. Keep the detector-sized boxes and only dedupe.
            filtered_results = _dedupe_candidates(filtered_results, min(cfg.nms_iou_threshold, 0.08))
        else:
            filtered_results = _normalize_candidate_boxes(filtered_results, model, image_bgr.shape[:2])
            filtered_results = _dedupe_candidates(filtered_results, min(cfg.nms_iou_threshold, 0.12))

    return filtered_results


def _post_filter_candidate(
    cand: _GroupCandidate,
    image_bgr: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> tuple[Candidate | None, dict[str, object]]:
    color_policy = _resolve_color_policy(model, cfg)
    simple_blob = _is_simple_blob_model(model)
    min_scale = _effective_min_scale(cfg, model)
    max_scale = _effective_max_scale(cfg, model)
    box_scale = 0.5 * (
        float(cand.w) / float(max(1, model.width))
        + float(cand.h) / float(max(1, model.height))
    )

    row: dict[str, object] = {
        "x": cand.x,
        "y": cand.y,
        "w": cand.w,
        "h": cand.h,
        "base_score": cand.score,
        "scale": cand.scale,
        "box_scale": box_scale,
        "effective_min_scale": min_scale,
        "effective_max_scale": max_scale,
    }

    y0, x0 = cand.y, cand.x
    y1 = min(image_bgr.shape[0], y0 + cand.h)
    x1 = min(image_bgr.shape[1], x0 + cand.w)
    if y1 <= y0 or x1 <= x0:
        row["decision"] = "out_of_bounds"
        return None, row

    cand_bgr = image_bgr[y0:y1, x0:x1]
    color_family_ok = _color_family_gate(cand_bgr, model, cfg)
    row["color_family_ok"] = color_family_ok
    if not color_family_ok:
        row["decision"] = "color_family_gate"
        return None, row

    center_ok = True
    if simple_blob:
        center_ok = _blob_center_gate(cand_bgr, model)
        row["center_ok"] = center_ok
        if not center_ok:
            row["decision"] = "blob_center_gate"
            return None, row
        if box_scale < min_scale or box_scale > max_scale:
            row["decision"] = "scale_out_of_range"
            return None, row

    color_sim = _color_similarity(cand_bgr, model, cfg) if (color_policy.weight > 0.0 or simple_blob) else 1.0
    min_color_similarity = color_policy.min_similarity if color_policy.weight > 0.0 else 0.0
    acceptance_threshold = cfg.match_threshold
    if simple_blob:
        min_color_similarity = max(0.14, color_policy.min_similarity - 0.10)
        acceptance_threshold = max(0.46, cfg.match_threshold)

    row["color_sim"] = color_sim
    row["min_color_similarity"] = min_color_similarity
    row["acceptance_threshold"] = acceptance_threshold

    if color_policy.weight > 0.0 and color_sim < min_color_similarity:
        row["decision"] = "color_similarity_gate"
        return None, row

    shape_sim = ""
    size_consistency = ""
    final_score = cand.score
    if simple_blob:
        shape_sim = _shape_similarity(
            Candidate(x=cand.x, y=cand.y, w=cand.w, h=cand.h, score=cand.score, angle=0.0, scale=box_scale),
            cand_bgr,
            model,
        )
        size_consistency = float(np.exp(-0.65 * abs(np.log(max(box_scale, 1e-6)))))
        row["shape_sim"] = shape_sim
        row["size_consistency"] = size_consistency
        if shape_sim < 0.14:
            row["decision"] = "shape_similarity_gate"
            return None, row

        color_mix = 0.18 + 0.28 * color_policy.weight
        shape_mix = 0.22
        size_mix = 0.14
        base_mix = max(0.0, 1.0 - color_mix - shape_mix - size_mix)
        final_score = (
            base_mix * cand.score
            + color_mix * color_sim
            + shape_mix * float(shape_sim)
            + size_mix * float(size_consistency)
        )
    elif color_policy.weight > 0.0:
        color_mix = 0.15 + 0.40 * color_policy.weight
        final_score = (1.0 - color_mix) * cand.score + color_mix * color_sim

    row["final_score"] = final_score
    if final_score < acceptance_threshold:
        row["decision"] = "score_below_threshold"
        return None, row

    row["decision"] = "accepted"
    return (
        Candidate(
            x=max(0, cand.x),
            y=max(0, cand.y),
            w=cand.w,
            h=cand.h,
            score=float(final_score),
            angle=0.0,
            scale=cand.scale,
        ),
        row,
    )


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
    binary = np.where(gray < threshold, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _prepare_template_image(template_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
    hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    strong_color = sat >= max(42.0, float(np.percentile(sat, 82)))
    dark_or_mid = val <= max(210.0, float(np.percentile(val, 88)))
    strong_mask = (strong_color & dark_or_mid).astype(np.uint8) * 255
    colored_pixels = int(np.count_nonzero(strong_mask))
    total_pixels = max(1, template_bgr.shape[0] * template_bgr.shape[1])
    colored_ratio = colored_pixels / float(total_pixels)
    sat95 = float(np.percentile(sat, 95))

    ring_like = False
    if colored_pixels > 0:
        comps = _components_from_binary(strong_mask, min_area=max(4, int(round(total_pixels * 0.01))))
        if comps:
            comp = max(comps, key=lambda item: item.area)
            coverage = float(comp.area) / float(max(1, comp.w * comp.h))
            center_dx = abs((comp.cx / float(max(1, template_bgr.shape[1]))) - 0.5)
            center_dy = abs((comp.cy / float(max(1, template_bgr.shape[0]))) - 0.5)
            centered = center_dx <= 0.18 and center_dy <= 0.18
            ring_like = coverage < 0.30 or not centered

    ignore_high_saturation = colored_ratio >= 0.06 and sat95 >= 70.0 and ring_like
    if not ignore_high_saturation:
        return template_bgr, False
    return _neutralize_high_saturation(template_bgr, sat_threshold=52.0, val_threshold=230.0), True
def _neutralize_high_saturation(image_bgr: np.ndarray, sat_threshold: float, val_threshold: float) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    strong_mask = (sat >= sat_threshold) & (val <= val_threshold)
    if not np.any(strong_mask):
        return image_bgr

    background = _estimate_border_background(image_bgr)
    neutralized = image_bgr.copy()
    neutralized[strong_mask] = background.astype(np.uint8)
    return neutralized


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


def _template_color_binary(image_bgr: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> np.ndarray:
    score = _color_response_map(image_bgr, model, cfg)
    threshold = _resolve_color_policy(model, cfg).binary_threshold
    binary = np.where(score >= threshold, 255, 0).astype(np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _clahe_binary(image_bgr: np.ndarray) -> np.ndarray:
    """CLAHE-equalized Otsu threshold — helps on faded/low-contrast scans."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    _, binary = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))


def _template_match_candidates(
    image_bgr: np.ndarray,
    tmpl_mask: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[_GroupCandidate]:
    """Multi-scale cv2.matchTemplate sweep — pixel-level correlation on grayscale or color-distance map."""
    color_policy = _resolve_color_policy(model, cfg)
    if color_policy.use_color_map:
        gray = np.clip(_color_response_map(image_bgr, model, cfg) * 255.0, 0, 255).astype(np.uint8)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ih, iw = gray.shape[:2]
    candidates: list[_GroupCandidate] = []

    effective_min_scale = _effective_min_scale(cfg, model)
    tpl_threshold = max(0.28, cfg.match_threshold - 0.10)
    scales = np.geomspace(effective_min_scale, _effective_max_scale(cfg, model), _effective_num_scales(cfg, model))

    for scale in scales:
        tw = max(4, int(round(model.width * scale)))
        th = max(4, int(round(model.height * scale)))
        if tw >= iw or th >= ih:
            continue

        tmpl_resized = cv2.resize(tmpl_mask, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray, tmpl_resized, cv2.TM_CCOEFF_NORMED)
        locs = np.argwhere(result >= tpl_threshold)
        for row, col in locs:
            base_score = float(result[row, col])
            candidates.append(
                _GroupCandidate(
                    x=int(col),
                    y=int(row),
                    w=tw,
                    h=th,
                    score=base_score,
                    scale=float(scale),
                    component_ids=(),
                )
            )

    return candidates


def _edge_match_candidates(
    image_bgr: np.ndarray,
    tmpl_mask: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[_GroupCandidate]:
    """Canny-edge matchTemplate — effective on line-art / CAD drawings."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    page_edges = cv2.Canny(gray, 40, 120)
    ih, iw = page_edges.shape[:2]
    candidates: list[_GroupCandidate] = []

    tpl_threshold = max(0.25, cfg.match_threshold - 0.15)
    scales = np.linspace(_effective_min_scale(cfg, model), _effective_max_scale(cfg, model), max(6, cfg.num_scales // 2))

    for scale in scales:
        tw = max(4, int(round(model.width * scale)))
        th = max(4, int(round(model.height * scale)))
        if tw >= iw or th >= ih:
            continue

        tmpl_resized = cv2.resize(tmpl_mask, (tw, th), interpolation=cv2.INTER_AREA)
        tmpl_edges = cv2.Canny(tmpl_resized, 40, 120)
        if not np.any(tmpl_edges):
            continue

        result = cv2.matchTemplate(page_edges, tmpl_edges, cv2.TM_CCOEFF_NORMED)
        locs = np.argwhere(result >= tpl_threshold)
        for row, col in locs:
            base_score = float(result[row, col]) * 0.90  # slight deflate so shape-matcher wins ties
            candidates.append(
                _GroupCandidate(
                    x=int(col),
                    y=int(row),
                    w=tw,
                    h=th,
                    score=base_score,
                    scale=float(scale),
                    component_ids=(),
                )
            )

    return candidates


def _simple_blob_candidates(
    image_bgr: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[_GroupCandidate]:
    if not _is_simple_blob_model(model):
        return []

    color_policy = _resolve_color_policy(model, cfg)
    binary = cv2.bitwise_or(
        _foreground_binary(image_bgr, dark_threshold=cfg.dark_threshold, for_template=False),
        _background_distance_binary(image_bgr),
    )
    if color_policy.use_color_map:
        binary = cv2.bitwise_or(binary, _template_color_binary(image_bgr, model, cfg))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    components = _components_from_binary(binary, min_area=cfg.min_component_area)
    if not components:
        return []

    effective_min_scale = _effective_min_scale(cfg, model)
    effective_max_scale = _effective_max_scale(cfg, model)
    min_area = max(cfg.min_component_area, int(round(model.area * (effective_min_scale**2) * 0.35)))
    max_area = max(cfg.max_component_area, int(round(model.area * (effective_max_scale**2) * 1.2)))
    min_score = max(0.30, cfg.match_threshold - 0.12)

    candidates: list[_GroupCandidate] = []
    for comp in components:
        if comp.area < min_area or comp.area > max_area:
            continue
        if comp.w < 3 or comp.h < 3:
            continue

        scale = float(np.sqrt(comp.area / float(max(1, model.area))))
        if scale < effective_min_scale or scale > effective_max_scale:
            continue

        circularity = _mask_circularity(comp.mask)
        solidity = _mask_solidity(comp.mask)
        circ_sim = max(0.0, 1.0 - abs(circularity - model.circularity) / 0.35)
        aspect_sim = _ratio_similarity(float(comp.w) / float(max(1, comp.h)), model.aspect_ratio)
        patch = image_bgr[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w]
        if not _color_family_gate(patch, model, cfg):
            continue
        if not _blob_center_gate(patch, model):
            continue
        color_sim = _color_similarity(patch, model, cfg)
        if color_sim < color_policy.min_similarity:
            continue

        solidity_sim = max(0.0, 1.0 - abs(solidity - model.solidity) / 0.28)
        shape_score = 0.56 * circ_sim + 0.20 * aspect_sim + 0.24 * solidity_sim
        color_mix = 0.10 + 0.42 * color_policy.weight
        score = (1.0 - color_mix) * shape_score + color_mix * color_sim
        if score < min_score:
            continue

        candidates.append(
            _GroupCandidate(
                x=comp.x,
                y=comp.y,
                w=comp.w,
                h=comp.h,
                score=float(score),
                scale=float(scale),
                component_ids=(comp.id,),
            )
        )

    return candidates


def _dense_blob_candidates(
    image_bgr: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[_GroupCandidate]:
    if not _is_simple_blob_model(model):
        return []

    response = _color_response_map(image_bgr, model, cfg)
    profile = model.color_profile
    policy = _resolve_color_policy(model, cfg)

    effective_min_scale = _effective_min_scale(cfg, model)
    effective_max_scale = _effective_max_scale(cfg, model)
    min_area = max(2, int(round(model.area * (effective_min_scale**2) * 0.28)))
    max_area = max(cfg.max_component_area, int(round(model.area * (effective_max_scale**2) * 1.35)))
    candidates: list[_GroupCandidate] = []

    thresholds = [max(0.16, policy.binary_threshold - 0.20), max(0.20, policy.binary_threshold - 0.14), max(0.24, policy.binary_threshold - 0.08)]
    if profile.chromatic:
        thresholds = [min(th, 0.30) for th in thresholds]

    for level_idx, base_threshold in enumerate(sorted(set(round(th, 3) for th in thresholds))):
        binary = np.where(response >= base_threshold, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        components = _components_from_binary(binary, min_area=max(2, cfg.min_component_area))
        if not components:
            continue

        for comp in components:
            if comp.area < min_area or comp.area > max_area:
                continue
            if comp.w < 3 or comp.h < 3:
                continue

            patch = image_bgr[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w]
            if not _color_family_gate(patch, model, cfg):
                continue
            if not _blob_center_gate(patch, model):
                continue
            color_sim = _color_similarity(patch, model, cfg)
            if color_sim < max(0.12, policy.min_similarity - 0.14):
                continue

            circularity = _mask_circularity(comp.mask)
            solidity = _mask_solidity(comp.mask)
            aspect_sim = _ratio_similarity(float(comp.w) / float(max(1, comp.h)), model.aspect_ratio)
            circ_sim = max(0.0, 1.0 - abs(circularity - model.circularity) / 0.58)
            solidity_sim = max(0.0, 1.0 - abs(solidity - model.solidity) / 0.46)

            mean_response = float(np.mean(response[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w][comp.mask > 0]))
            scale = float(np.sqrt(comp.area / float(max(1, model.area))))
            if scale < effective_min_scale or scale > effective_max_scale:
                continue

            size_consistency = float(np.exp(-0.5 * abs(np.log(max(scale, 1e-6)))))
            score = (
                0.36 * color_sim
                + 0.18 * circ_sim
                + 0.14 * solidity_sim
                + 0.08 * aspect_sim
                + 0.10 * size_consistency
                + 0.14 * mean_response
            )
            if score < max(0.22, cfg.match_threshold - 0.20):
                continue

            candidates.append(
                _GroupCandidate(
                    x=comp.x,
                    y=comp.y,
                    w=comp.w,
                    h=comp.h,
                    score=float(score),
                    scale=float(scale),
                    component_ids=(1000000 * (level_idx + 1) + comp.id,),
                )
            )

    return candidates


def _centered_blob_proposals(
    image_bgr: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> list[_GroupCandidate]:
    if not _is_simple_blob_model(model):
        return []

    response = _color_response_map(image_bgr, model, cfg)
    response_blur = cv2.GaussianBlur(response, (0, 0), sigmaX=1.0, sigmaY=1.0)
    peak_floor = max(0.14, float(np.percentile(response_blur, 98.8)))

    spacing = max(3, int(round(max(model.width, model.height) * 1.0)))
    kernel = np.ones((spacing * 2 + 1, spacing * 2 + 1), dtype=np.uint8)
    local_max = cv2.dilate(response_blur, kernel)
    peak_mask = (response_blur >= peak_floor) & (response_blur >= local_max - 1e-6)
    peak_points = list(zip(*np.where(peak_mask)))
    if not peak_points:
        return []

    policy = _resolve_color_policy(model, cfg)
    eff_min_scale = _effective_min_scale(cfg, model)
    eff_max_scale = _effective_max_scale(cfg, model)
    min_area = max(2, int(round(model.area * (eff_min_scale**2) * 0.36)))
    max_area = max(8, int(round(model.area * (eff_max_scale**2) * 1.45)))

    binary_levels = [
        max(0.10, policy.binary_threshold - 0.28),
        max(0.14, policy.binary_threshold - 0.22),
        max(0.18, policy.binary_threshold - 0.16),
        max(0.22, policy.binary_threshold - 0.10),
    ]
    ih, iw = image_bgr.shape[:2]
    candidates: list[_GroupCandidate] = []

    for idx, (cy, cx) in enumerate(peak_points, start=1):
        peak_score = float(response_blur[cy, cx])
        best_candidate = None
        best_score = 0.0

        for level_idx, level in enumerate(binary_levels, start=1):
            candidate = _candidate_from_peak(
                image_bgr=image_bgr,
                response=response_blur,
                model=model,
                cfg=cfg,
                peak_x=int(cx),
                peak_y=int(cy),
                threshold=float(level),
                min_area=min_area,
                max_area=max_area,
                component_token=2000000 + idx * 10 + level_idx,
            )
            if candidate is None:
                continue
            if candidate.score > best_score:
                best_score = candidate.score
                best_candidate = candidate

        if best_candidate is not None and best_candidate.score >= max(0.14, cfg.match_threshold - 0.22):
            candidates.append(best_candidate)

    return candidates


def _candidate_from_peak(
    image_bgr: np.ndarray,
    response: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
    peak_x: int,
    peak_y: int,
    threshold: float,
    min_area: int,
    max_area: int,
    component_token: int,
) -> _GroupCandidate | None:
    binary = np.where(response >= threshold, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if peak_y < 0 or peak_y >= labels.shape[0] or peak_x < 0 or peak_x >= labels.shape[1]:
        return None
    label_id = int(labels[peak_y, peak_x])
    if label_id <= 0 or label_id >= count:
        return None

    x, y, w, h, area = stats[label_id]
    if area < min_area or area > max_area:
        return None
    if w < 3 or h < 3:
        return None

    patch = image_bgr[y:y + h, x:x + w]
    if patch.size == 0:
        return None
    if not _color_family_gate(patch, model, cfg):
        return None
    if not _blob_center_gate(patch, model):
        return None

    color_sim = _color_similarity(patch, model, cfg)
    if color_sim < 0.06:
        return None

    scale = float(np.sqrt(area / float(max(1, model.area))))
    if scale < _effective_min_scale(cfg, model) or scale > _effective_max_scale(cfg, model):
        return None

    shape_sim = _shape_similarity(
        Candidate(x=int(x), y=int(y), w=int(w), h=int(h), score=0.0, angle=0.0, scale=scale),
        patch,
        model,
    )
    if shape_sim < 0.04:
        return None

    component_mask = np.where(labels[y:y + h, x:x + w] == label_id, 255, 0).astype(np.uint8)
    circularity = _mask_circularity(component_mask)
    solidity = _mask_solidity(component_mask)
    circ_sim = max(0.0, 1.0 - abs(circularity - model.circularity) / 0.62)
    solidity_sim = max(0.0, 1.0 - abs(solidity - model.solidity) / 0.52)
    aspect_sim = _ratio_similarity(float(w) / float(max(1, h)), model.aspect_ratio)
    center_peak = float(response[peak_y, peak_x])
    size_consistency = float(np.exp(-0.55 * abs(np.log(max(scale, 1e-6)))))

    score = (
        0.22 * color_sim
        + 0.20 * shape_sim
        + 0.14 * circ_sim
        + 0.10 * solidity_sim
        + 0.08 * aspect_sim
        + 0.10 * size_consistency
        + 0.16 * center_peak
    )
    return _GroupCandidate(
        x=int(x),
        y=int(y),
        w=int(w),
        h=int(h),
        score=float(score),
        scale=float(scale),
        component_ids=(component_token,),
    )


def _estimate_border_background(image_bgr: np.ndarray) -> np.ndarray:
    top = image_bgr[0, :, :]
    bottom = image_bgr[-1, :, :]
    left = image_bgr[:, 0, :]
    right = image_bgr[:, -1, :]
    border = np.concatenate((top, bottom, left, right), axis=0)
    return np.median(border.astype(np.float32), axis=0)


def _auto_dark_threshold(gray: np.ndarray) -> int:
    percentile = float(np.percentile(gray, 0.35))
    return int(np.clip(percentile + 18.0, 70.0, 145.0))


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

    effective_min_scale = _effective_min_scale(cfg, model)
    effective_max_scale = _effective_max_scale(cfg, model)
    min_area = max(cfg.min_component_area, int(round(model.area * (effective_min_scale**2) * 0.03)))
    max_area = max(cfg.max_component_area, int(round(model.area * (effective_max_scale**2) * 1.4)))
    min_height = max(cfg.min_component_height, int(round(model.height * effective_min_scale * 0.18)))
    max_height = max(min_height + 1, int(round(model.height * effective_max_scale * 1.3)))
    max_width = max(cfg.max_component_width, int(round(model.width * effective_max_scale * 1.3)))

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
    image_bgr: np.ndarray,
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

        candidate = _score_group(group, image_bgr, model, cfg)
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

    max_span = max(model.width, model.height) * _effective_max_scale(cfg, model) * 2.4
    for idx, anchor in enumerate(components):
        neighbors: list[_Component] = []
        for other in components[idx + 1 :]:
            if other.x - anchor.x > max_span:
                break
            if abs(other.cy - anchor.cy) > max(anchor.h, other.h, model.height * _effective_max_scale(cfg, model)):
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
    image_bgr: np.ndarray,
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
    if scale < _effective_min_scale(cfg, model) or scale > _effective_max_scale(cfg, model):
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
    layout_sim = _layout_similarity(group, x0, y0, width, height, union_area, model)
    color_policy = _resolve_color_policy(model, cfg)
    patch = image_bgr[y0:y1, x0:x1]
    if not _color_family_gate(patch, model, cfg):
        return None
    color_sim = _color_similarity(patch, model, cfg)
    if color_sim < color_policy.min_similarity:
        return None

    base_score = (
        0.38 * iou
        + 0.20 * dilated_iou
        + 0.20 * hu_sim
        + 0.10 * aspect_sim
        + 0.12 * layout_sim
    )
    color_mix = 0.12 + 0.35 * color_policy.weight
    score = (1.0 - color_mix) * base_score + color_mix * color_sim
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


def _mask_circularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0
    return float(np.clip((4.0 * np.pi * area) / (perimeter * perimeter), 0.0, 1.0))


def _mask_solidity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 0.0:
        return 0.0
    pixel_area = float(np.count_nonzero(mask))
    return float(np.clip(pixel_area / hull_area, 0.0, 1.0))


def _effective_scale_bounds(cfg: DetectionConfig, model: _TemplateModel) -> tuple[float, float]:
    min_scale = float(cfg.min_scale)
    max_scale = float(cfg.max_scale)
    if _is_simple_blob_model(model):
        # Tiny filled markups are very sensitive to rasterization.
        # Give them a little headroom above 1.0 so real dots are not rejected,
        # while still keeping a meaningful floor against tiny text specks.
        min_dim = float(max(1, min(model.width, model.height)))
        max_dim = float(max(1, max(model.width, model.height)))
        min_scale = max(min_scale, max(0.22, min(0.45, 3.2 / max_dim)))
        max_scale = max(max_scale, 1.0 + min(0.22, 2.0 / min_dim))
        if cfg.uniform_size_assist:
            min_scale = max(min_scale, 0.62)
            max_scale = max(max_scale, 1.28)
    if max_scale < min_scale:
        max_scale = min_scale
    return min_scale, max_scale


def _effective_min_scale(cfg: DetectionConfig, model: _TemplateModel) -> float:
    return _effective_scale_bounds(cfg, model)[0]


def _effective_max_scale(cfg: DetectionConfig, model: _TemplateModel) -> float:
    return _effective_scale_bounds(cfg, model)[1]


def _effective_num_scales(cfg: DetectionConfig, model: _TemplateModel) -> int:
    if _is_simple_blob_model(model):
        return max(cfg.num_scales, 24)
    return cfg.num_scales


def _is_simple_blob_model(model: _TemplateModel) -> bool:
    return (
        model.component_count == 1
        and model.circularity >= 0.56
        and model.solidity >= 0.55
        and 0.62 <= model.aspect_ratio <= 1.55
    )


def _resolve_color_policy(model: _TemplateModel, cfg: DetectionConfig) -> _ColorPolicy:
    mode = cfg.color_sensitivity.strip().lower()
    if mode not in {"auto", "soft", "strict"}:
        mode = "auto"

    base_weight = float(np.clip(model.color_weight, 0.0, 1.0))
    if mode == "soft":
        weight = base_weight * 0.55
        tolerance_scale = 1.35
        min_similarity = 0.14 + 0.22 * weight
        binary_threshold = 0.30 + 0.16 * weight
    elif mode == "strict":
        floor = 0.65 if model.color_profile.chromatic else 0.30
        weight = max(base_weight, floor)
        tolerance_scale = 0.78
        min_similarity = 0.26 + 0.34 * weight
        binary_threshold = 0.42 + 0.22 * weight
    else:
        weight = base_weight
        tolerance_scale = 1.0
        if model.color_profile.chromatic:
            min_similarity = 0.18 + 0.28 * weight
            binary_threshold = 0.34 + 0.20 * weight
        else:
            min_similarity = 0.14 + 0.24 * weight
            binary_threshold = 0.28 + 0.14 * weight

    return _ColorPolicy(
        weight=float(np.clip(weight, 0.0, 1.0)),
        min_similarity=float(np.clip(min_similarity, 0.08, 0.72)),
        binary_threshold=float(np.clip(binary_threshold, 0.22, 0.82)),
        tolerance_scale=float(np.clip(tolerance_scale, 0.65, 1.60)),
        use_color_map=weight >= 0.12 and model.color_profile.chromatic,
    )


def _template_color_signature(
    template_bgr: np.ndarray,
    mask: np.ndarray,
) -> tuple[_ColorProfile, float]:
    fg = mask > 0
    if not np.any(fg):
        neutral = _ColorProfile(
            fg_lab_mean=(128.0, 128.0, 128.0),
            fg_lab_std=(0.0, 0.0, 0.0),
            fg_hue_mean=0.0,
            fg_hue_spread=180.0,
            fg_sat_mean=0.0,
            fg_sat_std=0.0,
            fg_chroma_mean=0.0,
            fg_chroma_std=0.0,
            chromatic=False,
        )
        return neutral, 0.0

    profile = _color_profile_from_mask(template_bgr, fg)
    color_weight = float(
        np.clip(
            max(
                profile.fg_sat_mean / 90.0,
                profile.fg_chroma_mean / 18.0,
            ) - 0.12,
            0.0,
            1.0,
        )
    )
    if not profile.chromatic:
        lightness_anchor = float(np.clip(abs(profile.fg_lab_mean[0] - 245.0) / 180.0, 0.0, 1.0))
        neutral_weight = 0.18 + 0.12 * lightness_anchor
        color_weight = max(color_weight * 0.35, neutral_weight)
    return profile, color_weight


def _color_profile_from_mask(image_bgr: np.ndarray, fg: np.ndarray) -> _ColorProfile:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    chroma = np.linalg.norm(lab[..., 1:] - np.asarray([128.0, 128.0], dtype=np.float32), axis=2)

    fg_lab = lab[fg]
    fg_h = hsv[..., 0][fg]
    fg_s = hsv[..., 1][fg]
    fg_c = chroma[fg]

    fg_lab_mean = tuple(float(v) for v in fg_lab.mean(axis=0))
    fg_lab_std = tuple(float(v) for v in fg_lab.std(axis=0))
    fg_sat_mean = float(fg_s.mean()) if fg_s.size else 0.0
    fg_sat_std = float(fg_s.std()) if fg_s.size else 0.0
    fg_chroma_mean = float(fg_c.mean()) if fg_c.size else 0.0
    fg_chroma_std = float(fg_c.std()) if fg_c.size else 0.0
    fg_hue_mean = _circular_hue_mean(fg_h)
    fg_hue_spread = _circular_hue_spread(fg_h, fg_hue_mean)
    chromatic = bool(fg_sat_mean >= 18.0 or fg_chroma_mean >= 10.0)

    return _ColorProfile(
        fg_lab_mean=fg_lab_mean,
        fg_lab_std=fg_lab_std,
        fg_hue_mean=fg_hue_mean,
        fg_hue_spread=fg_hue_spread,
        fg_sat_mean=fg_sat_mean,
        fg_sat_std=fg_sat_std,
        fg_chroma_mean=fg_chroma_mean,
        fg_chroma_std=fg_chroma_std,
        chromatic=chromatic,
    )


def _color_response_map(image_bgr: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> np.ndarray:
    policy = _resolve_color_policy(model, cfg)
    profile = model.color_profile
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    chroma = np.linalg.norm(lab[..., 1:] - np.asarray([128.0, 128.0], dtype=np.float32), axis=2)

    fg_mean = np.asarray(profile.fg_lab_mean, dtype=np.float32)
    fg_std = np.asarray(profile.fg_lab_std, dtype=np.float32)
    lab_tol = np.maximum(fg_std * 2.8 + 8.0, np.asarray([10.0, 8.0, 8.0], dtype=np.float32)) * policy.tolerance_scale

    fg_dist = np.sqrt(np.mean(((lab - fg_mean) / lab_tol) ** 2, axis=2))
    lab_score = np.exp(-(fg_dist**2))

    if profile.chromatic:
        hue_diff = _circular_hue_diff(hsv[..., 0], profile.fg_hue_mean)
        hue_tol = max(10.0, profile.fg_hue_spread * 2.4 + 10.0) * policy.tolerance_scale
        sat_tol = max(12.0, profile.fg_sat_std * 2.8 + 12.0) * policy.tolerance_scale
        chroma_tol = max(8.0, profile.fg_chroma_std * 2.6 + 8.0) * policy.tolerance_scale
        hue_score = np.exp(-((hue_diff / hue_tol) ** 2))
        sat_score = np.exp(-(((hsv[..., 1] - profile.fg_sat_mean) / sat_tol) ** 2))
        chroma_score = np.exp(-(((chroma - profile.fg_chroma_mean) / chroma_tol) ** 2))
        sat_level = np.clip(hsv[..., 1] / max(profile.fg_sat_mean, 1.0), 0.0, 1.0)
        chroma_level = np.clip(chroma / max(profile.fg_chroma_mean, 1.0), 0.0, 1.0)
        presence_gate = 0.25 + 0.75 * (0.5 * sat_level + 0.5 * chroma_level)
        score = (0.52 * lab_score + 0.20 * hue_score + 0.14 * sat_score + 0.14 * chroma_score) * presence_gate
    else:
        l_tol = max(10.0, profile.fg_lab_std[0] * 2.8 + 10.0) * policy.tolerance_scale
        chroma_tol = max(10.0, profile.fg_chroma_std * 3.0 + 10.0) * policy.tolerance_scale
        light_score = np.exp(-(((lab[..., 0] - profile.fg_lab_mean[0]) / l_tol) ** 2))
        chroma_score = np.exp(-(((chroma - profile.fg_chroma_mean) / chroma_tol) ** 2))
        score = 0.74 * lab_score + 0.16 * light_score + 0.10 * chroma_score

    return np.clip(score.astype(np.float32), 0.0, 1.0)


def _extract_patch_color_profile(candidate_bgr: np.ndarray, model: _TemplateModel) -> _ColorProfile:
    if _is_simple_blob_model(model):
        fg = _candidate_foreground_mask(candidate_bgr, model)
        return _color_profile_from_mask(candidate_bgr, fg)

    resized = cv2.resize(candidate_bgr, (model.width, model.height), interpolation=cv2.INTER_AREA)
    fg = model.mask > 0
    return _color_profile_from_mask(resized, fg)


def _candidate_foreground_mask(candidate_bgr: np.ndarray, model: _TemplateModel | None = None) -> np.ndarray:
    if model is not None and model.color_profile.chromatic:
        cfg = DetectionConfig(
            color_sensitivity="soft",
            chromatic_gate_on=0.35,
            hue_gate_max_diff=max(32.0, DetectionConfig().hue_gate_max_diff),
        )
        response = _color_response_map(candidate_bgr, model, cfg)
        thresh = max(0.20, _resolve_color_policy(model, cfg).binary_threshold - 0.16)
        fg = response >= thresh
        if np.count_nonzero(fg) >= max(6, int(round(candidate_bgr.shape[0] * candidate_bgr.shape[1] * 0.08))):
            return fg

    bg_color = _estimate_border_background(candidate_bgr)
    color_dist = np.linalg.norm(candidate_bgr.astype(np.float32) - bg_color, axis=2)
    gray = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = gray <= min(235, int(np.percentile(gray, 85)))
    color_mask = color_dist >= max(6.0, float(np.percentile(color_dist, 65)))
    fg = dark_mask | color_mask
    if not np.any(fg):
        fg = np.ones(candidate_bgr.shape[:2], dtype=bool)
    return fg


def _color_similarity(candidate_bgr: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> float:
    if candidate_bgr.size == 0:
        return 0.0

    policy = _resolve_color_policy(model, cfg)
    if policy.weight <= 0.0:
        return 1.0

    candidate = _extract_patch_color_profile(candidate_bgr, model)
    profile = model.color_profile

    lab_tol = np.maximum(
        np.asarray(profile.fg_lab_std, dtype=np.float32) * 3.2 + 10.0,
        np.asarray([12.0, 10.0, 10.0], dtype=np.float32),
    ) * policy.tolerance_scale
    fg_delta = np.asarray(candidate.fg_lab_mean, dtype=np.float32) - np.asarray(profile.fg_lab_mean, dtype=np.float32)
    lab_score = np.exp(-np.mean((fg_delta / lab_tol) ** 2))

    if profile.chromatic:
        hue_tol = max(14.0, profile.fg_hue_spread * 2.8 + 14.0) * policy.tolerance_scale
        sat_tol = max(18.0, profile.fg_sat_std * 3.0 + 18.0) * policy.tolerance_scale
        chroma_tol = max(12.0, profile.fg_chroma_std * 3.0 + 12.0) * policy.tolerance_scale
        hue_score = np.exp(-(_circular_hue_diff(candidate.fg_hue_mean, profile.fg_hue_mean) / hue_tol) ** 2)
        sat_score = np.exp(-((candidate.fg_sat_mean - profile.fg_sat_mean) / sat_tol) ** 2)
        chroma_score = np.exp(-((candidate.fg_chroma_mean - profile.fg_chroma_mean) / chroma_tol) ** 2)
        sat_level = np.clip(candidate.fg_sat_mean / max(profile.fg_sat_mean, 1.0), 0.0, 1.0)
        chroma_level = np.clip(candidate.fg_chroma_mean / max(profile.fg_chroma_mean, 1.0), 0.0, 1.0)
        presence_gate = 0.55 + 0.45 * (0.45 * sat_level + 0.55 * chroma_level)
        score = (0.42 * lab_score + 0.24 * hue_score + 0.17 * sat_score + 0.17 * chroma_score) * presence_gate
    else:
        l_tol = max(10.0, profile.fg_lab_std[0] * 2.8 + 10.0) * policy.tolerance_scale
        chroma_tol = max(10.0, profile.fg_chroma_std * 3.0 + 10.0) * policy.tolerance_scale
        light_score = np.exp(-((candidate.fg_lab_mean[0] - profile.fg_lab_mean[0]) / l_tol) ** 2)
        chroma_score = np.exp(-((candidate.fg_chroma_mean - profile.fg_chroma_mean) / chroma_tol) ** 2)
        score = 0.74 * lab_score + 0.16 * light_score + 0.10 * chroma_score

    return float(np.clip(score, 0.0, 1.0))


def _color_family_gate(candidate_bgr: np.ndarray, model: _TemplateModel, cfg: DetectionConfig) -> bool:
    if candidate_bgr.size == 0:
        return False
    profile = model.color_profile
    if not profile.chromatic:
        return True

    sat_ref = profile.fg_sat_mean / 255.0
    chroma_ref = profile.fg_chroma_mean / 120.0
    chroma_conf = float(np.clip(0.55 * sat_ref + 0.45 * chroma_ref, 0.0, 1.0))
    if chroma_conf < cfg.chromatic_gate_on:
        return True

    candidate = _extract_patch_color_profile(candidate_bgr, model)
    hue_diff = float(_circular_hue_diff(candidate.fg_hue_mean, profile.fg_hue_mean))
    hue_limit = max(12.0, cfg.hue_gate_max_diff, profile.fg_hue_spread * 2.2 + 12.0)
    sat_floor = max(10.0, profile.fg_sat_mean * 0.24)
    chroma_floor = max(6.0, profile.fg_chroma_mean * 0.22)
    lab_delta = np.abs(
        np.asarray(candidate.fg_lab_mean, dtype=np.float32)
        - np.asarray(profile.fg_lab_mean, dtype=np.float32)
    )
    lab_ok = bool(
        lab_delta[0] <= max(18.0, profile.fg_lab_std[0] * 3.2 + 18.0)
        and lab_delta[1] <= max(16.0, profile.fg_lab_std[1] * 3.2 + 16.0)
        and lab_delta[2] <= max(16.0, profile.fg_lab_std[2] * 3.2 + 16.0)
    )
    chroma_ok = candidate.fg_sat_mean >= sat_floor and candidate.fg_chroma_mean >= chroma_floor
    if hue_diff <= hue_limit and chroma_ok:
        return True
    return lab_ok and candidate.fg_chroma_mean >= max(4.0, chroma_floor * 0.7)


def _shape_similarity(candidate: Candidate, patch: np.ndarray, model: _TemplateModel) -> float:
    if patch.size == 0:
        return 0.0
    if _is_simple_blob_model(model):
        mask = _dominant_blob_mask(patch, model)
        if mask is None:
            return 0.0
        resized = cv2.resize(mask, (model.width, model.height), interpolation=cv2.INTER_NEAREST)
        iou = _mask_iou(resized, model.mask)
        circularity = _mask_circularity(mask)
        solidity = _mask_solidity(mask)
        circ_sim = max(0.0, 1.0 - abs(circularity - model.circularity) / 0.45)
        solidity_sim = max(0.0, 1.0 - abs(solidity - model.solidity) / 0.34)
        aspect = float(candidate.w) / float(max(1, candidate.h))
        aspect_sim = _ratio_similarity(aspect, model.aspect_ratio)
        size_scale = 0.5 * (
            float(candidate.w) / float(max(1, model.width))
            + float(candidate.h) / float(max(1, model.height))
        )
        size_consistency = float(np.exp(-0.7 * abs(np.log(max(size_scale, 1e-6)))))
        return float(0.34 * iou + 0.24 * circ_sim + 0.18 * solidity_sim + 0.10 * aspect_sim + 0.14 * size_consistency)

    resized = cv2.resize(patch, (model.width, model.height), interpolation=cv2.INTER_AREA)
    mask = _foreground_binary(resized, dark_threshold=0, for_template=True)
    iou = _mask_iou(mask, model.mask)
    hu_sim = _similarity_from_distance(_hu_distance(_hu_signature(mask), model.hu), scale=2.5)
    aspect = float(candidate.w) / float(max(1, candidate.h))
    aspect_sim = _ratio_similarity(aspect, model.aspect_ratio)
    size_scale = 0.5 * (float(candidate.w) / float(max(1, model.width)) + float(candidate.h) / float(max(1, model.height)))
    size_consistency = float(np.exp(-0.7 * abs(np.log(max(size_scale, 1e-6)))))
    return float(0.40 * iou + 0.25 * hu_sim + 0.15 * aspect_sim + 0.20 * size_consistency)


def _blob_center_gate(candidate_bgr: np.ndarray, model: _TemplateModel) -> bool:
    if model.color_profile.chromatic and not _center_response_gate(candidate_bgr, model):
        return False
    mask = _dominant_blob_mask(candidate_bgr, model)
    if mask is None:
        return False

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] <= 0.0:
        return False

    h, w = mask.shape[:2]
    cx = float(moments["m10"] / moments["m00"])
    cy = float(moments["m01"] / moments["m00"])
    dx = abs(cx - (w / 2.0)) / float(max(1.0, w / 2.0))
    dy = abs(cy - (h / 2.0)) / float(max(1.0, h / 2.0))
    if dx > 0.72 or dy > 0.72:
        return False

    area_ratio = float(np.count_nonzero(mask)) / float(max(1, w * h))
    if area_ratio < 0.04:
        return False

    circularity = _mask_circularity(mask)
    solidity = _mask_solidity(mask)
    return circularity >= max(0.22, model.circularity - 0.46) and solidity >= max(0.30, model.solidity - 0.42)


def _center_response_gate(candidate_bgr: np.ndarray, model: _TemplateModel) -> bool:
    cfg = DetectionConfig(color_sensitivity="soft", chromatic_gate_on=0.35, hue_gate_max_diff=32.0)
    response = _color_response_map(candidate_bgr, model, cfg)
    if response.size == 0:
        return False
    peak_index = int(np.argmax(response))
    peak_y, peak_x = np.unravel_index(peak_index, response.shape)
    h, w = response.shape[:2]
    dx = abs(float(peak_x) - (w / 2.0)) / float(max(1.0, w / 2.0))
    dy = abs(float(peak_y) - (h / 2.0)) / float(max(1.0, h / 2.0))
    peak_value = float(response[peak_y, peak_x])
    return peak_value >= 0.10 and dx <= 0.90 and dy <= 0.90


def _dominant_blob_mask(candidate_bgr: np.ndarray, model: _TemplateModel) -> np.ndarray | None:
    fg = _candidate_foreground_mask(candidate_bgr, model)
    binary = np.where(fg, 255, 0).astype(np.uint8)
    components = _components_from_binary(binary, min_area=2)
    if not components:
        return None

    h, w = binary.shape[:2]
    cx_mid = w / 2.0
    cy_mid = h / 2.0
    max_dist = max(1.0, float(np.hypot(cx_mid, cy_mid)))
    best_comp: _Component | None = None
    best_score = -1.0
    for comp in components:
        dist = float(np.hypot(comp.cx - cx_mid, comp.cy - cy_mid)) / max_dist
        center_score = max(0.0, 1.0 - dist)
        area_score = min(1.0, comp.area / float(max(1, model.area)))
        circularity = _mask_circularity(comp.mask)
        solidity = _mask_solidity(comp.mask)
        score = 0.42 * center_score + 0.26 * area_score + 0.18 * circularity + 0.14 * solidity
        if score > best_score:
            best_score = score
            best_comp = comp

    if best_comp is None:
        return None
    return best_comp.mask


def _merge_multi_class_candidates(
    page_img: np.ndarray,
    models: Sequence[_TemplateModel],
    per_class_candidates: Sequence[Sequence[Candidate]],
    cfg: DetectionConfig,
) -> tuple[list[Candidate], list[int], int]:
    class_counts = [0 for _ in models]
    assigned: list[_AssignedCandidate] = []
    unclassified = 0

    for class_idx, candidates in enumerate(per_class_candidates):
        for cand in candidates:
            patch = page_img[cand.y : cand.y + cand.h, cand.x : cand.x + cand.w]
            if _text_like_rejection_gate(patch):
                continue
            class_scores: list[float] = []
            shape_scores: list[float] = []
            for model in models:
                if not _size_gate_for_class(cand, model, cfg):
                    class_scores.append(0.0)
                    shape_scores.append(0.0)
                    continue
                if not _color_family_gate(patch, model, cfg):
                    class_scores.append(0.0)
                    shape_scores.append(0.0)
                    continue
                color_sim = _color_similarity(patch, model, cfg)
                shape_sim = _shape_similarity(cand, patch, model)
                min_shape_gate = 0.20
                if _is_simple_blob_model(model):
                    min_shape_gate = 0.12
                if shape_sim < min_shape_gate:
                    class_scores.append(0.0)
                    shape_scores.append(shape_sim)
                    continue
                shape_scores.append(shape_sim)
                if _is_simple_blob_model(model):
                    class_scores.append(0.64 * color_sim + 0.36 * shape_sim)
                else:
                    class_scores.append(0.50 * color_sim + 0.50 * shape_sim)

            best_idx = int(np.argmax(class_scores))
            best_score = float(class_scores[best_idx])
            if best_score <= 0.0:
                continue
            sorted_scores = sorted(class_scores, reverse=True)
            second = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
            margin = best_score - second
            if margin < cfg.class_assignment_margin:
                if cfg.exclude_unclassified:
                    unclassified += 1
                    continue
                best_idx = class_idx

            min_keep_shape = 0.30
            if _is_simple_blob_model(models[best_idx]):
                min_keep_shape = 0.16
            if shape_scores[best_idx] < min_keep_shape:
                continue

            scale = 0.5 * (
                float(cand.w) / float(max(1, models[best_idx].width))
                + float(cand.h) / float(max(1, models[best_idx].height))
            )
            size_consistency = float(np.exp(-abs(np.log(max(scale, 1e-6)))))
            assigned.append(
                _AssignedCandidate(
                    class_index=best_idx,
                    candidate=Candidate(
                        x=cand.x,
                        y=cand.y,
                        w=cand.w,
                        h=cand.h,
                        score=best_score,
                        angle=cand.angle,
                        scale=scale,
                    ),
                    margin=margin,
                    score=best_score,
                    size_consistency=size_consistency,
                )
            )

    if cfg.global_non_overlap:
        assigned = _global_non_overlap_assigned(assigned)

    merged: list[Candidate] = []
    for item in assigned:
        class_counts[item.class_index] += 1
        merged.append(item.candidate)
    return merged, class_counts, unclassified


def _size_gate_for_class(cand: Candidate, model: _TemplateModel, cfg: DetectionConfig) -> bool:
    width_scale = float(cand.w) / float(max(1, model.width))
    height_scale = float(cand.h) / float(max(1, model.height))
    scale = 0.5 * (width_scale + height_scale)
    min_scale = _effective_min_scale(cfg, model)
    max_scale = _effective_max_scale(cfg, model)
    if cfg.uniform_size_assist and not _is_simple_blob_model(model):
        min_scale = max(min_scale, 0.60)
        max_scale = min(max_scale, 1.50)
    anisotropy = max(width_scale, height_scale) / max(1e-6, min(width_scale, height_scale))
    return min_scale <= scale <= max_scale and anisotropy <= 1.35


def _text_like_rejection_gate(patch: np.ndarray) -> bool:
    if patch.size == 0:
        return True
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    comps = _components_from_binary(binary, min_area=1)
    if not comps:
        return False

    small_bits = [comp for comp in comps if comp.area <= 18]
    tiny_ratio = len(small_bits) / float(max(1, len(comps)))
    max_area = max(comp.area for comp in comps)
    patch_area = max(1, patch.shape[0] * patch.shape[1])
    dominant_ratio = max_area / float(patch_area)

    # Text patches usually have many tiny fragments and no dominant body.
    return len(comps) >= 5 and tiny_ratio >= 0.65 and dominant_ratio <= 0.22


def _global_non_overlap_assigned(items: Sequence[_AssignedCandidate]) -> list[_AssignedCandidate]:
    ordered = sorted(items, key=lambda item: (item.margin, item.score, item.size_consistency), reverse=True)
    kept: list[_AssignedCandidate] = []
    for item in ordered:
        cand = item.candidate
        if any(_candidate_boxes_overlap(cand, prev.candidate) for prev in kept):
            continue
        kept.append(item)
    return kept


def _circular_hue_mean(values) -> float:
    if np.size(values) == 0:
        return 0.0
    angles = np.asarray(values, dtype=np.float32) * (2.0 * np.pi / 180.0)
    sin_mean = float(np.mean(np.sin(angles)))
    cos_mean = float(np.mean(np.cos(angles)))
    angle = np.arctan2(sin_mean, cos_mean)
    if angle < 0.0:
        angle += 2.0 * np.pi
    return float(angle * (180.0 / (2.0 * np.pi)))


def _circular_hue_spread(values, mean: float) -> float:
    if np.size(values) == 0:
        return 180.0
    diffs = _circular_hue_diff(np.asarray(values, dtype=np.float32), mean)
    return float(np.mean(diffs))


def _circular_hue_diff(values, mean: float) -> np.ndarray | float:
    arr = np.asarray(values, dtype=np.float32)
    diff = np.abs(arr - float(mean))
    diff = np.minimum(diff, 180.0 - diff)
    if np.isscalar(values):
        return float(diff)
    return diff


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


def _normalize_candidate_boxes(
    candidates: Sequence[Candidate],
    model: _TemplateModel,
    image_shape: tuple[int, int],
) -> list[Candidate]:
    if not candidates:
        return []

    target_scale = _weighted_median_scale(candidates)
    target_w = max(2, int(round(model.width * target_scale)))
    target_h = max(2, int(round(model.height * target_scale)))
    image_h, image_w = image_shape

    normalized: list[Candidate] = []
    for cand in candidates:
        cx = cand.x + cand.w / 2.0
        cy = cand.y + cand.h / 2.0
        x = int(round(cx - target_w / 2.0))
        y = int(round(cy - target_h / 2.0))
        x = int(np.clip(x, 0, max(0, image_w - target_w)))
        y = int(np.clip(y, 0, max(0, image_h - target_h)))
        normalized.append(
            Candidate(
                x=x,
                y=y,
                w=target_w,
                h=target_h,
                score=cand.score,
                angle=cand.angle,
                scale=target_scale,
            )
        )

    return normalized


def _weighted_median_scale(candidates: Sequence[Candidate]) -> float:
    ordered = sorted((cand.scale, max(cand.score, 1e-6)) for cand in candidates)
    total_weight = sum(weight for _, weight in ordered)
    threshold = total_weight * 0.5
    running = 0.0
    for scale, weight in ordered:
        running += weight
        if running >= threshold:
            return float(scale)
    return float(ordered[-1][0])


def _dedupe_candidates(candidates: Sequence[Candidate], iou_threshold: float) -> list[Candidate]:
    ordered = sorted(candidates, key=lambda cand: cand.score, reverse=True)
    kept: list[Candidate] = []

    for cand in ordered:
        if any(
            _candidate_iou(cand, prev) >= iou_threshold
            or (_candidate_boxes_overlap(cand, prev) and _candidate_center_too_close(cand, prev))
            for prev in kept
        ):
            continue
        kept.append(cand)

    return kept


def _candidate_iou(a: Candidate, b: Candidate) -> float:
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


def _candidate_boxes_overlap(a: Candidate, b: Candidate) -> bool:
    return not (
        a.x + a.w <= b.x
        or b.x + b.w <= a.x
        or a.y + a.h <= b.y
        or b.y + b.h <= a.y
    )


def _candidate_center_too_close(a: Candidate, b: Candidate) -> bool:
    ax = a.x + a.w / 2.0
    ay = a.y + a.h / 2.0
    bx = b.x + b.w / 2.0
    by = b.y + b.h / 2.0
    dist = float(np.hypot(ax - bx, ay - by))
    min_dim = float(min(a.w, a.h, b.w, b.h))
    return dist < 0.72 * min_dim


def _resolve_scope_polygon(
    image_shape: tuple[int, int] | tuple[int, int, int],
    scope: tuple[tuple[float, float], ...] | None,
) -> tuple[tuple[int, int], ...] | None:
    if scope is None:
        return None
    height, width = image_shape[:2]
    polygon = tuple(
        (
            int(round(np.clip(point[0], 0.0, 1.0) * width)),
            int(round(np.clip(point[1], 0.0, 1.0) * height)),
        )
        for point in scope
    )
    if len(polygon) < 3:
        return None
    return polygon


def _polygon_bbox(polygon: tuple[tuple[int, int], ...] | None) -> tuple[int, int, int, int] | None:
    if polygon is None:
        return None
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x0 = min(xs)
    y0 = min(ys)
    x1 = max(xs)
    y1 = max(ys)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _offset_candidates(candidates: Sequence[Candidate], dx: int, dy: int) -> list[Candidate]:
    if dx == 0 and dy == 0:
        return list(candidates)
    return [
        Candidate(
            x=cand.x + dx,
            y=cand.y + dy,
            w=cand.w,
            h=cand.h,
            score=cand.score,
            angle=cand.angle,
            scale=cand.scale,
        )
        for cand in candidates
    ]


def _filter_candidates_by_scope(
    candidates: Sequence[Candidate],
    scope_polygon: tuple[tuple[int, int], ...] | None,
    min_overlap: float,
) -> list[Candidate]:
    if scope_polygon is None:
        return list(candidates)

    mask = _polygon_mask(scope_polygon)
    mask_h, mask_w = mask.shape[:2]
    kept: list[Candidate] = []
    for cand in candidates:
        xx1 = max(cand.x, 0)
        yy1 = max(cand.y, 0)
        xx2 = min(cand.x + cand.w, mask_w)
        yy2 = min(cand.y + cand.h, mask_h)
        if xx2 <= xx1 or yy2 <= yy1:
            continue
        overlap = int(np.count_nonzero(mask[yy1:yy2, xx1:xx2]))
        area = max(1, (xx2 - xx1) * (yy2 - yy1))
        if float(overlap) / float(area) >= min_overlap:
            kept.append(cand)
    return kept


def _polygon_mask(polygon: tuple[tuple[int, int], ...]) -> np.ndarray:
    bbox = _polygon_bbox(polygon)
    if bbox is None:
        return np.zeros((1, 1), dtype=np.uint8)
    _, _, x1, y1 = bbox
    mask = np.zeros((y1 + 2, x1 + 2), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], 255)
    return mask


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


def _draw_candidates(
    image: np.ndarray,
    candidates: Sequence[Candidate],
    scope_polygon: tuple[tuple[int, int], ...] | None = None,
) -> np.ndarray:
    annotated = image.copy()
    if scope_polygon is not None:
        pts = np.asarray(scope_polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 180, 0), thickness=4)
        x0, y0, _, _ = _polygon_bbox(scope_polygon)
        cv2.putText(
            annotated,
            "Scope",
            (x0 + 6, max(24, y0 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 140, 0),
            2,
            cv2.LINE_AA,
        )

    for idx, cand in enumerate(candidates, start=1):
        cv2.rectangle(
            annotated,
            (cand.x, cand.y),
            (cand.x + cand.w, cand.y + cand.h),
            (0, 255, 0),
            3,
        )
        label = str(idx)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(text_h + 8, cand.y)
        cv2.rectangle(
            annotated,
            (cand.x, top - text_h - baseline - 8),
            (cand.x + text_w + 12, top),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (cand.x + 6, top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
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


def _debug_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\t", " ").replace("\n", " ")


def _write_debug_table(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_debug_text(row.get(col, "")) for col in columns))
    path.write_text("\n".join(lines), encoding="utf-8")


def _trace_simple_blob_route(
    image_bgr: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> tuple[list[_GroupCandidate], list[dict[str, object]], dict[str, object]]:
    summary: dict[str, object] = {
        "simple_blob_model": _is_simple_blob_model(model),
    }
    if not _is_simple_blob_model(model):
        return [], [], summary

    color_policy = _resolve_color_policy(model, cfg)
    binary = cv2.bitwise_or(
        _foreground_binary(image_bgr, dark_threshold=cfg.dark_threshold, for_template=False),
        _background_distance_binary(image_bgr),
    )
    if color_policy.use_color_map:
        binary = cv2.bitwise_or(binary, _template_color_binary(image_bgr, model, cfg))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    components = _components_from_binary(binary, min_area=cfg.min_component_area)

    effective_min_scale = _effective_min_scale(cfg, model)
    effective_max_scale = _effective_max_scale(cfg, model)
    min_area = max(cfg.min_component_area, int(round(model.area * (effective_min_scale**2) * 0.35)))
    max_area = max(cfg.max_component_area, int(round(model.area * (effective_max_scale**2) * 1.2)))
    min_score = max(0.30, cfg.match_threshold - 0.12)
    summary.update(
        {
            "simple_blob_components_total": len(components),
            "simple_blob_min_area": min_area,
            "simple_blob_max_area": max_area,
            "simple_blob_min_scale": effective_min_scale,
            "simple_blob_max_scale": effective_max_scale,
            "simple_blob_min_score": min_score,
        }
    )

    accepted: list[_GroupCandidate] = []
    rows: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}

    for comp in components:
        row: dict[str, object] = {
            "route": "simple_blob",
            "component_id": comp.id,
            "x": comp.x,
            "y": comp.y,
            "w": comp.w,
            "h": comp.h,
            "area": comp.area,
        }
        decision = "accepted"
        scale = float(np.sqrt(comp.area / float(max(1, model.area))))
        row["scale"] = scale

        if comp.area < min_area or comp.area > max_area:
            decision = "area_out_of_range"
        elif comp.w < 3 or comp.h < 3:
            decision = "too_small"
        elif scale < effective_min_scale or scale > effective_max_scale:
            decision = "scale_out_of_range"
        else:
            patch = image_bgr[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w]
            color_family_ok = _color_family_gate(patch, model, cfg)
            center_ok = _blob_center_gate(patch, model) if color_family_ok else False
            color_sim = _color_similarity(patch, model, cfg) if center_ok else 0.0
            circularity = _mask_circularity(comp.mask)
            solidity = _mask_solidity(comp.mask)
            circ_sim = max(0.0, 1.0 - abs(circularity - model.circularity) / 0.35)
            aspect_sim = _ratio_similarity(float(comp.w) / float(max(1, comp.h)), model.aspect_ratio)
            solidity_sim = max(0.0, 1.0 - abs(solidity - model.solidity) / 0.28)
            shape_score = 0.56 * circ_sim + 0.20 * aspect_sim + 0.24 * solidity_sim
            color_mix = 0.10 + 0.42 * color_policy.weight
            score = (1.0 - color_mix) * shape_score + color_mix * color_sim
            row.update(
                {
                    "color_family_ok": color_family_ok,
                    "center_ok": center_ok,
                    "color_sim": color_sim,
                    "min_color_similarity": color_policy.min_similarity,
                    "circularity": circularity,
                    "solidity": solidity,
                    "circ_sim": circ_sim,
                    "aspect_sim": aspect_sim,
                    "solidity_sim": solidity_sim,
                    "shape_score": shape_score,
                    "score": score,
                }
            )

            if not color_family_ok:
                decision = "color_family_gate"
            elif not center_ok:
                decision = "blob_center_gate"
            elif color_sim < color_policy.min_similarity:
                decision = "color_similarity_gate"
            elif score < min_score:
                decision = "score_below_threshold"
            else:
                accepted.append(
                    _GroupCandidate(
                        x=comp.x,
                        y=comp.y,
                        w=comp.w,
                        h=comp.h,
                        score=float(score),
                        scale=float(scale),
                        component_ids=(comp.id,),
                    )
                )

        row["decision"] = decision
        reason_counts[decision] = reason_counts.get(decision, 0) + 1
        rows.append(row)

    summary["simple_blob_accepted"] = len(accepted)
    for reason, count in sorted(reason_counts.items()):
        summary[f"simple_blob_{reason}"] = count
    return accepted, rows, summary


def _trace_dense_blob_route(
    image_bgr: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> tuple[list[_GroupCandidate], list[dict[str, object]], dict[str, object]]:
    summary: dict[str, object] = {}
    if not _is_simple_blob_model(model):
        return [], [], summary

    response = _color_response_map(image_bgr, model, cfg)
    profile = model.color_profile
    policy = _resolve_color_policy(model, cfg)
    effective_min_scale = _effective_min_scale(cfg, model)
    effective_max_scale = _effective_max_scale(cfg, model)
    min_area = max(2, int(round(model.area * (effective_min_scale**2) * 0.28)))
    max_area = max(cfg.max_component_area, int(round(model.area * (effective_max_scale**2) * 1.35)))
    thresholds = [max(0.16, policy.binary_threshold - 0.20), max(0.20, policy.binary_threshold - 0.14), max(0.24, policy.binary_threshold - 0.08)]
    if profile.chromatic:
        thresholds = [min(th, 0.30) for th in thresholds]
    thresholds = sorted(set(round(th, 3) for th in thresholds))

    summary.update(
        {
            "dense_blob_thresholds": ",".join(f"{th:.3f}" for th in thresholds),
            "dense_blob_min_area": min_area,
            "dense_blob_max_area": max_area,
            "dense_blob_min_scale": effective_min_scale,
            "dense_blob_max_scale": effective_max_scale,
        }
    )

    accepted: list[_GroupCandidate] = []
    rows: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}
    total_components = 0

    for level_idx, base_threshold in enumerate(thresholds, start=1):
        binary = np.where(response >= base_threshold, 255, 0).astype(np.uint8)
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        components = _components_from_binary(binary, min_area=max(2, cfg.min_component_area))
        total_components += len(components)
        summary[f"dense_blob_level{level_idx}_components"] = len(components)

        for comp in components:
            row: dict[str, object] = {
                "route": "dense_blob",
                "level_idx": level_idx,
                "threshold": base_threshold,
                "component_id": comp.id,
                "x": comp.x,
                "y": comp.y,
                "w": comp.w,
                "h": comp.h,
                "area": comp.area,
            }
            decision = "accepted"
            patch = image_bgr[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w]
            scale = float(np.sqrt(comp.area / float(max(1, model.area))))
            row["scale"] = scale

            if comp.area < min_area or comp.area > max_area:
                decision = "area_out_of_range"
            elif comp.w < 3 or comp.h < 3:
                decision = "too_small"
            else:
                color_family_ok = _color_family_gate(patch, model, cfg)
                center_ok = _blob_center_gate(patch, model) if color_family_ok else False
                color_sim = _color_similarity(patch, model, cfg) if center_ok else 0.0
                circularity = _mask_circularity(comp.mask)
                solidity = _mask_solidity(comp.mask)
                aspect_sim = _ratio_similarity(float(comp.w) / float(max(1, comp.h)), model.aspect_ratio)
                circ_sim = max(0.0, 1.0 - abs(circularity - model.circularity) / 0.58)
                solidity_sim = max(0.0, 1.0 - abs(solidity - model.solidity) / 0.46)
                mean_response = float(np.mean(response[comp.y:comp.y + comp.h, comp.x:comp.x + comp.w][comp.mask > 0]))
                size_consistency = float(np.exp(-0.5 * abs(np.log(max(scale, 1e-6)))))
                score = (
                    0.36 * color_sim
                    + 0.18 * circ_sim
                    + 0.14 * solidity_sim
                    + 0.08 * aspect_sim
                    + 0.10 * size_consistency
                    + 0.14 * mean_response
                )
                row.update(
                    {
                        "color_family_ok": color_family_ok,
                        "center_ok": center_ok,
                        "color_sim": color_sim,
                        "min_color_similarity": max(0.12, policy.min_similarity - 0.14),
                        "circularity": circularity,
                        "solidity": solidity,
                        "circ_sim": circ_sim,
                        "aspect_sim": aspect_sim,
                        "solidity_sim": solidity_sim,
                        "mean_response": mean_response,
                        "size_consistency": size_consistency,
                        "score": score,
                    }
                )

                if not color_family_ok:
                    decision = "color_family_gate"
                elif not center_ok:
                    decision = "blob_center_gate"
                elif color_sim < max(0.12, policy.min_similarity - 0.14):
                    decision = "color_similarity_gate"
                elif scale < effective_min_scale or scale > effective_max_scale:
                    decision = "scale_out_of_range"
                elif score < max(0.22, cfg.match_threshold - 0.20):
                    decision = "score_below_threshold"
                else:
                    accepted.append(
                        _GroupCandidate(
                            x=comp.x,
                            y=comp.y,
                            w=comp.w,
                            h=comp.h,
                            score=float(score),
                            scale=float(scale),
                            component_ids=(1000000 * level_idx + comp.id,),
                        )
                    )

            row["decision"] = decision
            reason_counts[decision] = reason_counts.get(decision, 0) + 1
            rows.append(row)

    summary["dense_blob_components_total"] = total_components
    summary["dense_blob_accepted"] = len(accepted)
    for reason, count in sorted(reason_counts.items()):
        summary[f"dense_blob_{reason}"] = count
    return accepted, rows, summary


def _candidate_from_peak_trace(
    image_bgr: np.ndarray,
    response: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
    peak_x: int,
    peak_y: int,
    threshold: float,
    min_area: int,
    max_area: int,
    component_token: int,
    peak_index: int,
    level_idx: int,
) -> tuple[_GroupCandidate | None, dict[str, object]]:
    row: dict[str, object] = {
        "route": "centered_blob",
        "peak_index": peak_index,
        "level_idx": level_idx,
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_score": float(response[peak_y, peak_x]) if 0 <= peak_y < response.shape[0] and 0 <= peak_x < response.shape[1] else 0.0,
        "threshold": threshold,
    }

    binary = np.where(response >= threshold, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if peak_y < 0 or peak_y >= labels.shape[0] or peak_x < 0 or peak_x >= labels.shape[1]:
        row["decision"] = "peak_out_of_bounds"
        return None, row
    label_id = int(labels[peak_y, peak_x])
    if label_id <= 0 or label_id >= count:
        row["decision"] = "no_component_at_peak"
        return None, row

    x, y, w, h, area = stats[label_id]
    row.update({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area)})
    if area < min_area or area > max_area:
        row["decision"] = "area_out_of_range"
        return None, row
    if w < 3 or h < 3:
        row["decision"] = "too_small"
        return None, row

    patch = image_bgr[y:y + h, x:x + w]
    if patch.size == 0:
        row["decision"] = "empty_patch"
        return None, row
    color_family_ok = _color_family_gate(patch, model, cfg)
    row["color_family_ok"] = color_family_ok
    if not color_family_ok:
        row["decision"] = "color_family_gate"
        return None, row
    center_ok = _blob_center_gate(patch, model)
    row["center_ok"] = center_ok
    if not center_ok:
        row["decision"] = "blob_center_gate"
        return None, row

    color_sim = _color_similarity(patch, model, cfg)
    row["color_sim"] = color_sim
    if color_sim < 0.06:
        row["decision"] = "color_similarity_gate"
        return None, row

    scale = float(np.sqrt(area / float(max(1, model.area))))
    row["scale"] = scale
    if scale < _effective_min_scale(cfg, model) or scale > _effective_max_scale(cfg, model):
        row["decision"] = "scale_out_of_range"
        return None, row

    shape_sim = _shape_similarity(
        Candidate(x=int(x), y=int(y), w=int(w), h=int(h), score=0.0, angle=0.0, scale=scale),
        patch,
        model,
    )
    row["shape_sim"] = shape_sim
    if shape_sim < 0.04:
        row["decision"] = "shape_similarity_gate"
        return None, row

    component_mask = np.where(labels[y:y + h, x:x + w] == label_id, 255, 0).astype(np.uint8)
    circularity = _mask_circularity(component_mask)
    solidity = _mask_solidity(component_mask)
    circ_sim = max(0.0, 1.0 - abs(circularity - model.circularity) / 0.62)
    solidity_sim = max(0.0, 1.0 - abs(solidity - model.solidity) / 0.52)
    aspect_sim = _ratio_similarity(float(w) / float(max(1, h)), model.aspect_ratio)
    center_peak = float(response[peak_y, peak_x])
    size_consistency = float(np.exp(-0.55 * abs(np.log(max(scale, 1e-6)))))
    score = (
        0.22 * color_sim
        + 0.20 * shape_sim
        + 0.14 * circ_sim
        + 0.10 * solidity_sim
        + 0.08 * aspect_sim
        + 0.10 * size_consistency
        + 0.16 * center_peak
    )
    row.update(
        {
            "circularity": circularity,
            "solidity": solidity,
            "circ_sim": circ_sim,
            "solidity_sim": solidity_sim,
            "aspect_sim": aspect_sim,
            "size_consistency": size_consistency,
            "score": score,
        }
    )
    row["decision"] = "accepted"
    return (
        _GroupCandidate(
            x=int(x),
            y=int(y),
            w=int(w),
            h=int(h),
            score=float(score),
            scale=float(scale),
            component_ids=(component_token,),
        ),
        row,
    )


def _trace_centered_blob_route(
    image_bgr: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
) -> tuple[list[_GroupCandidate], list[dict[str, object]], dict[str, object]]:
    summary: dict[str, object] = {}
    if not _is_simple_blob_model(model):
        return [], [], summary

    response = _color_response_map(image_bgr, model, cfg)
    response_blur = cv2.GaussianBlur(response, (0, 0), sigmaX=1.0, sigmaY=1.0)
    peak_floor = max(0.14, float(np.percentile(response_blur, 98.8)))
    spacing = max(3, int(round(max(model.width, model.height) * 1.0)))
    kernel = np.ones((spacing * 2 + 1, spacing * 2 + 1), dtype=np.uint8)
    local_max = cv2.dilate(response_blur, kernel)
    peak_mask = (response_blur >= peak_floor) & (response_blur >= local_max - 1e-6)
    peak_points = list(zip(*np.where(peak_mask)))

    policy = _resolve_color_policy(model, cfg)
    eff_min_scale = _effective_min_scale(cfg, model)
    eff_max_scale = _effective_max_scale(cfg, model)
    min_area = max(2, int(round(model.area * (eff_min_scale**2) * 0.36)))
    max_area = max(8, int(round(model.area * (eff_max_scale**2) * 1.45)))
    binary_levels = [
        max(0.10, policy.binary_threshold - 0.28),
        max(0.14, policy.binary_threshold - 0.22),
        max(0.18, policy.binary_threshold - 0.16),
        max(0.22, policy.binary_threshold - 0.10),
    ]
    summary.update(
        {
            "centered_blob_peak_floor": peak_floor,
            "centered_blob_peak_count": len(peak_points),
            "centered_blob_thresholds": ",".join(f"{th:.3f}" for th in binary_levels),
            "centered_blob_min_scale": eff_min_scale,
            "centered_blob_max_scale": eff_max_scale,
        }
    )

    accepted: list[_GroupCandidate] = []
    rows: list[dict[str, object]] = []
    reason_counts: dict[str, int] = {}

    for peak_index, (cy, cx) in enumerate(peak_points, start=1):
        best_candidate: _GroupCandidate | None = None
        best_score = -1.0
        best_row: dict[str, object] | None = None
        accepted_here = False
        for level_idx, level in enumerate(binary_levels, start=1):
            candidate, row = _candidate_from_peak_trace(
                image_bgr=image_bgr,
                response=response_blur,
                model=model,
                cfg=cfg,
                peak_x=int(cx),
                peak_y=int(cy),
                threshold=float(level),
                min_area=min_area,
                max_area=max_area,
                component_token=2000000 + peak_index * 10 + level_idx,
                peak_index=peak_index,
                level_idx=level_idx,
            )
            rows.append(row)
            decision = str(row.get("decision", "unknown"))
            reason_counts[decision] = reason_counts.get(decision, 0) + 1
            if candidate is not None and candidate.score > best_score:
                best_candidate = candidate
                best_score = candidate.score
                best_row = row
            if candidate is not None and candidate.score >= max(0.14, cfg.match_threshold - 0.22):
                accepted_here = True

        if accepted_here and best_candidate is not None:
            accepted.append(best_candidate)
            if best_row is not None:
                best_row["best_for_peak"] = True

    summary["centered_blob_accepted"] = len(accepted)
    for reason, count in sorted(reason_counts.items()):
        summary[f"centered_blob_{reason}"] = count
    return accepted, rows, summary


def _nms_groups_trace(
    candidates: Sequence[_GroupCandidate],
    iou_threshold: float,
) -> tuple[set[int], dict[int, str]]:
    if not candidates:
        return set(), {}

    ordered = sorted(enumerate(candidates), key=lambda item: item[1].score, reverse=True)
    kept: list[tuple[int, _GroupCandidate]] = []
    kept_indices: set[int] = set()
    used_component_sets: set[tuple[int, ...]] = set()
    suppressed: dict[int, str] = {}

    for idx, cand in ordered:
        if cand.component_ids in used_component_sets:
            suppressed[idx] = "duplicate_component_ids"
            continue

        reason = ""
        for prev_idx, prev in kept:
            if _group_iou(cand, prev) >= iou_threshold:
                reason = f"iou_with_{prev_idx + 1}"
                break
            if _group_center_too_close(cand, prev):
                reason = f"center_close_to_{prev_idx + 1}"
                break
        if reason:
            suppressed[idx] = reason
            continue

        kept.append((idx, cand))
        kept_indices.add(idx)
        used_component_sets.add(cand.component_ids)

    return kept_indices, suppressed


def _candidate_matches_final(cand: _GroupCandidate, finals: Sequence[Candidate]) -> bool:
    for final in finals:
        x1 = max(cand.x, final.x)
        y1 = max(cand.y, final.y)
        x2 = min(cand.x + cand.w, final.x + final.w)
        y2 = min(cand.y + cand.h, final.y + final.h)
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter = inter_w * inter_h
        if inter == 0:
            continue
        union = cand.w * cand.h + final.w * final.h - inter
        if union > 0 and (inter / float(union)) >= 0.90:
            return True
    return False


def _export_template_debug(
    debug_dir: Path,
    class_idx: int,
    class_name: str,
    template_img: np.ndarray,
    model: _TemplateModel,
) -> None:
    safe_name = _safe_debug_name(class_name)
    prefix = f"class{class_idx:02d}_{safe_name}"
    _imwrite_unicode(debug_dir / f"{prefix}_template.png", template_img)
    mask_vis = np.where(model.mask > 0, 255, 0).astype(np.uint8)
    _imwrite_unicode(debug_dir / f"{prefix}_template_mask.png", mask_vis)
    (debug_dir / f"{prefix}_template_info.txt").write_text(
        "\n".join(
            [
                f"class_name={class_name}",
                f"width={model.width}",
                f"height={model.height}",
                f"area={model.area}",
                f"component_count={model.component_count}",
                f"circularity={model.circularity:.6f}",
                f"solidity={model.solidity:.6f}",
                f"aspect_ratio={model.aspect_ratio:.6f}",
                f"color_weight={model.color_weight:.6f}",
                f"ignore_high_saturation={model.ignore_high_saturation}",
                f"chromatic={model.color_profile.chromatic}",
                f"fg_lab_mean={model.color_profile.fg_lab_mean}",
                f"fg_lab_std={model.color_profile.fg_lab_std}",
                f"fg_hue_mean={model.color_profile.fg_hue_mean:.6f}",
                f"fg_hue_spread={model.color_profile.fg_hue_spread:.6f}",
                f"fg_sat_mean={model.color_profile.fg_sat_mean:.6f}",
                f"fg_sat_std={model.color_profile.fg_sat_std:.6f}",
                f"fg_chroma_mean={model.color_profile.fg_chroma_mean:.6f}",
                f"fg_chroma_std={model.color_profile.fg_chroma_std:.6f}",
            ]
        ),
        encoding="utf-8",
    )


def _export_page_debug(
    debug_dir: Path,
    page_idx: int,
    class_idx: int,
    class_name: str,
    detect_view: np.ndarray,
    detect_scale: float,
    crop_x: int,
    crop_y: int,
    page_img: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
    final_candidates: Sequence[Candidate],
) -> None:
    safe_name = _safe_debug_name(class_name)
    prefix = f"page{page_idx:03d}_class{class_idx:02d}_{safe_name}"
    response = _color_response_map(detect_view, model, cfg)
    response_vis = np.clip(response * 255.0, 0.0, 255.0).astype(np.uint8)
    response_vis = cv2.applyColorMap(response_vis, cv2.COLORMAP_TURBO)
    _imwrite_unicode(debug_dir / f"{prefix}_response.png", response_vis)

    simple_blob = _simple_blob_candidates(detect_view, model, cfg) if _is_simple_blob_model(model) else []
    dense_blob = _dense_blob_candidates(detect_view, model, cfg) if _is_simple_blob_model(model) else []
    centered_blob = _centered_blob_proposals(detect_view, model, cfg) if _is_simple_blob_model(model) else []
    _imwrite_unicode(debug_dir / f"{prefix}_simple_blob.png", _draw_group_candidates(detect_view, simple_blob, (0, 200, 255)))
    _imwrite_unicode(debug_dir / f"{prefix}_dense_blob.png", _draw_group_candidates(detect_view, dense_blob, (255, 180, 0)))
    _imwrite_unicode(debug_dir / f"{prefix}_centered_blob.png", _draw_group_candidates(detect_view, centered_blob, (180, 255, 0)))

    final_local = _scale_candidates(final_candidates, detect_scale)
    final_local = _offset_candidates(final_local, -crop_x, -crop_y)
    _imwrite_unicode(debug_dir / f"{prefix}_final.png", _draw_candidates(detect_view, final_local))

    policy = _resolve_color_policy(model, cfg)
    response_percentiles = np.percentile(response, [90, 95, 98, 99, 99.5]) if response.size else np.zeros(5, dtype=np.float32)
    summary_lines = [
        f"class_name={class_name}",
        f"simple_blob_model={_is_simple_blob_model(model)}",
        f"final_candidates={len(final_candidates)}",
        f"detect_scale={detect_scale:.6f}",
        f"crop_offset=({crop_x},{crop_y})",
        f"uniform_size_assist={cfg.uniform_size_assist}",
        f"policy_weight={policy.weight:.6f}",
        f"policy_min_similarity={policy.min_similarity:.6f}",
        f"policy_binary_threshold={policy.binary_threshold:.6f}",
        f"policy_tolerance_scale={policy.tolerance_scale:.6f}",
        f"effective_min_scale={_effective_min_scale(cfg, model):.6f}",
        f"effective_max_scale={_effective_max_scale(cfg, model):.6f}",
        f"response_p90={float(response_percentiles[0]):.6f}",
        f"response_p95={float(response_percentiles[1]):.6f}",
        f"response_p98={float(response_percentiles[2]):.6f}",
        f"response_p99={float(response_percentiles[3]):.6f}",
        f"response_p99_5={float(response_percentiles[4]):.6f}",
        f"response_max={float(np.max(response)):.6f}",
    ]

    if _is_simple_blob_model(model):
        summary_lines.extend(
            [
                f"simple_blob_candidates={len(simple_blob)}",
                f"dense_blob_candidates={len(dense_blob)}",
                f"centered_blob_candidates={len(centered_blob)}",
            ]
        )
        simple_trace_candidates, simple_rows, simple_summary = _trace_simple_blob_route(detect_view, model, cfg)
        dense_trace_candidates, dense_rows, dense_summary = _trace_dense_blob_route(detect_view, model, cfg)
        centered_trace_candidates, centered_rows, centered_summary = _trace_centered_blob_route(detect_view, model, cfg)

        _write_debug_table(
            debug_dir / f"{prefix}_simple_blob_trace.tsv",
            (
                "route", "component_id", "x", "y", "w", "h", "area", "scale",
                "color_family_ok", "center_ok", "color_sim", "min_color_similarity",
                "circularity", "solidity", "circ_sim", "aspect_sim", "solidity_sim",
                "shape_score", "score", "decision",
            ),
            simple_rows,
        )
        _write_debug_table(
            debug_dir / f"{prefix}_dense_blob_trace.tsv",
            (
                "route", "level_idx", "threshold", "component_id", "x", "y", "w", "h", "area", "scale",
                "color_family_ok", "center_ok", "color_sim", "min_color_similarity",
                "circularity", "solidity", "circ_sim", "aspect_sim", "solidity_sim",
                "mean_response", "size_consistency", "score", "decision",
            ),
            dense_rows,
        )
        _write_debug_table(
            debug_dir / f"{prefix}_centered_blob_trace.tsv",
            (
                "route", "peak_index", "level_idx", "peak_x", "peak_y", "peak_score", "threshold",
                "x", "y", "w", "h", "area", "scale", "color_family_ok", "center_ok",
                "color_sim", "shape_sim", "circularity", "solidity", "circ_sim",
                "solidity_sim", "aspect_sim", "size_consistency", "score", "decision", "best_for_peak",
            ),
            centered_rows,
        )

        accepted_sources: list[tuple[str, _GroupCandidate]] = []
        accepted_sources.extend(("simple_blob", cand) for cand in simple_trace_candidates)
        accepted_sources.extend(("dense_blob", cand) for cand in dense_trace_candidates)
        accepted_sources.extend(("centered_blob", cand) for cand in centered_trace_candidates)
        accepted_only = [cand for _, cand in accepted_sources]
        kept_indices, suppressed = _nms_groups_trace(accepted_only, cfg.nms_iou_threshold)
        accepted_rows: list[dict[str, object]] = []
        for idx, (route_name, cand) in enumerate(accepted_sources, start=1):
            accepted_rows.append(
                {
                    "proposal_index": idx,
                    "route": route_name,
                    "x": cand.x,
                    "y": cand.y,
                    "w": cand.w,
                    "h": cand.h,
                    "score": cand.score,
                    "scale": cand.scale,
                    "nms_status": "kept" if (idx - 1) in kept_indices else "suppressed",
                    "nms_reason": suppressed.get(idx - 1, ""),
                    "matches_final_output": _candidate_matches_final(cand, final_local),
                }
            )
        _write_debug_table(
            debug_dir / f"{prefix}_accepted_proposals.tsv",
            ("proposal_index", "route", "x", "y", "w", "h", "score", "scale", "nms_status", "nms_reason", "matches_final_output"),
            accepted_rows,
        )

        post_filter_rows: list[dict[str, object]] = []
        post_filter_reason_counts: dict[str, int] = {}
        for idx in sorted(kept_indices):
            route_name, cand = accepted_sources[idx]
            _, detail = _post_filter_candidate(cand, detect_view, model, cfg)
            decision = str(detail.get("decision", "unknown"))
            post_filter_reason_counts[decision] = post_filter_reason_counts.get(decision, 0) + 1
            post_filter_rows.append(
                {
                    "proposal_index": idx + 1,
                    "route": route_name,
                    "x": cand.x,
                    "y": cand.y,
                    "w": cand.w,
                    "h": cand.h,
                    "base_score": cand.score,
                    "scale": cand.scale,
                    "box_scale": detail.get("box_scale", ""),
                    "effective_min_scale": detail.get("effective_min_scale", ""),
                    "effective_max_scale": detail.get("effective_max_scale", ""),
                    "color_family_ok": detail.get("color_family_ok", ""),
                    "center_ok": detail.get("center_ok", ""),
                    "color_sim": detail.get("color_sim", ""),
                    "min_color_similarity": detail.get("min_color_similarity", ""),
                    "shape_sim": detail.get("shape_sim", ""),
                    "size_consistency": detail.get("size_consistency", ""),
                    "acceptance_threshold": detail.get("acceptance_threshold", ""),
                    "final_score": detail.get("final_score", ""),
                    "decision": decision,
                    "matches_final_output": _candidate_matches_final(cand, final_local),
                }
            )
        _write_debug_table(
            debug_dir / f"{prefix}_post_filter_trace.tsv",
            (
                "proposal_index", "route", "x", "y", "w", "h", "base_score", "scale", "box_scale",
                "effective_min_scale", "effective_max_scale", "color_family_ok", "center_ok",
                "color_sim", "min_color_similarity", "shape_sim", "size_consistency",
                "acceptance_threshold", "final_score", "decision", "matches_final_output",
            ),
            post_filter_rows,
        )

        summary_lines.extend(
            f"{key}={_debug_text(value)}"
            for source_summary in (simple_summary, dense_summary, centered_summary)
            for key, value in source_summary.items()
        )
        summary_lines.append(f"pre_nms_accepted_total={len(accepted_only)}")
        summary_lines.append(f"post_nms_kept_total={len(kept_indices)}")
        for reason, count in sorted(post_filter_reason_counts.items()):
            summary_lines.append(f"post_filter_{reason}={count}")
    else:
        summary_lines.extend(
            [
                f"simple_blob_candidates={len(simple_blob)}",
                f"dense_blob_candidates={len(dense_blob)}",
                f"centered_blob_candidates={len(centered_blob)}",
            ]
        )

    _write_debug_table(
        debug_dir / f"{prefix}_final_candidates.tsv",
        ("candidate_index", "x", "y", "w", "h", "score", "scale"),
        [
            {
                "candidate_index": idx,
                "x": cand.x,
                "y": cand.y,
                "w": cand.w,
                "h": cand.h,
                "score": cand.score,
                "scale": cand.scale,
            }
            for idx, cand in enumerate(final_local, start=1)
        ],
    )

    (debug_dir / f"{prefix}_counts.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )


def _draw_group_candidates(
    image: np.ndarray,
    candidates: Sequence[_GroupCandidate],
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = image.copy()
    for idx, cand in enumerate(candidates, start=1):
        cv2.rectangle(canvas, (cand.x, cand.y), (cand.x + cand.w, cand.y + cand.h), color, 2)
        cv2.putText(
            canvas,
            str(idx),
            (cand.x, max(14, cand.y - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _safe_debug_name(value: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return text or "markup"
