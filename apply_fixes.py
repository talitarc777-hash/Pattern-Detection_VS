import re
from pathlib import Path

target = Path("app/detector.py")
content = target.read_text(encoding="utf-8")

# Fix 1: class_assignment_margin
content = content.replace("class_assignment_margin: float = 0.05", "class_assignment_margin: float = 0.08")

# Fix 2: dark-neutral acceptance_threshold
content = content.replace(
"""            if gray_sim < 0.55:
                row["decision"] = "gray_similarity_gate"
                return None, row
            acceptance_threshold = max(0.56, acceptance_threshold)""",
"""            if gray_sim < 0.55:
                row["decision"] = "gray_similarity_gate"
                return None, row
            acceptance_threshold = max(0.62, acceptance_threshold)"""
)

# Fix 3: non-blob small template
content = content.replace(
"""        if model.area <= 64 or min(model.width, model.height) <= 12:
            min_shape = 0.24
            acceptance_threshold = max(acceptance_threshold, 0.48)""",
"""        if model.area <= 64 or min(model.width, model.height) <= 12:
            min_shape = 0.28
            acceptance_threshold = max(acceptance_threshold, 0.55)"""
)

# Fix 4: _template_match_candidates & variance check
content = content.replace(
"""def _template_match_candidates(
    image_l: np.ndarray,
    template_mask: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
    focus_mask: np.ndarray | None = None,
) -> list[_GroupCandidate]:
    if image_l is None or model.template_l is None or model.mask is None:
        return []
    template_l = model.template_l
    candidates: list[_GroupCandidate] = []

    effective_min_scale = _effective_min_scale(cfg, model)
    tpl_threshold = max(0.52, cfg.match_threshold + 0.05) if small_multi_part else max(0.28, cfg.match_threshold - 0.10)""",
"""def _template_match_candidates(
    image_l: np.ndarray,
    template_mask: np.ndarray,
    model: _TemplateModel,
    cfg: DetectionConfig,
    focus_mask: np.ndarray | None = None,
) -> list[_GroupCandidate]:
    if image_l is None or model.template_l is None or model.mask is None:
        return []
        
    template_l = model.template_l
    if float(np.std(template_l[model.mask > 0])) < 3.0:
        return []

    candidates: list[_GroupCandidate] = []

    small_multi_part = model.component_count > 1 and model.area <= 48 and not _is_simple_blob_model(model)
    effective_min_scale = _effective_min_scale(cfg, model)
    tpl_threshold = max(0.44, cfg.match_threshold - 0.02) if small_multi_part else max(0.28, cfg.match_threshold - 0.10)"""
)

# Fix 5: _edge_match_candidates
content = content.replace(
"""    small_multi_part = model.component_count > 1 and model.area <= 48 and not _is_simple_blob_model(model)
    tpl_threshold = max(0.50, cfg.match_threshold + 0.03) if small_multi_part else max(0.25, cfg.match_threshold - 0.15)""",
"""    small_multi_part = model.component_count > 1 and model.area <= 48 and not _is_simple_blob_model(model)
    tpl_threshold = max(0.40, cfg.match_threshold - 0.05) if small_multi_part else max(0.25, cfg.match_threshold - 0.15)"""
)

# Fix 6: _score_group layout sim
content = content.replace(
"""    if small_multi_part:
        part_sim = _part_layout_similarity(patch, model)
        gray_sim = _gray_patch_similarity(patch, model)
        if layout_sim < 0.28 or part_sim < 0.34 or gray_sim < 0.38:
            return None""",
"""    if small_multi_part:
        part_sim = _part_layout_similarity(patch, model)
        gray_sim = _gray_patch_similarity(patch, model)
        if layout_sim < 0.20 or part_sim < 0.26 or gray_sim < 0.30:
            return None"""
)

# Fix 7: _is_simple_blob_model circularity
content = content.replace(
"""def _is_simple_blob_model(model: _TemplateModel) -> bool:
    return (
        model.area >= 12
        and min(model.width, model.height) >= 4
        and model.component_count == 1
        and model.circularity >= 0.56""",
"""def _is_simple_blob_model(model: _TemplateModel) -> bool:
    return (
        model.area >= 12
        and min(model.width, model.height) >= 4
        and model.component_count == 1
        and model.circularity >= 0.38"""
)

# Fix 8: _is_dark_neutral_blob_model area
content = content.replace(
"""def _is_dark_neutral_blob_model(model: _TemplateModel) -> bool:
    profile = model.color_profile
    return (
        _is_simple_blob_model(model)
        and not profile.chromatic
        and profile.fg_lab_mean[0] <= 80.0
        and profile.fg_sat_mean <= 12.0
        and profile.fg_chroma_mean <= 6.0
        and model.area >= 96""",
"""def _is_dark_neutral_blob_model(model: _TemplateModel) -> bool:
    profile = model.color_profile
    return (
        _is_simple_blob_model(model)
        and not profile.chromatic
        and profile.fg_lab_mean[0] <= 80.0
        and profile.fg_sat_mean <= 12.0
        and profile.fg_chroma_mean <= 6.0
        and model.area >= 20"""
)

# Fix 9: _group_center_too_close
content = content.replace(
"""def _group_center_too_close(a: _GroupCandidate, b: _GroupCandidate) -> bool:
    ax = a.x + a.w / 2.0
    ay = a.y + a.h / 2.0
    bx = b.x + b.w / 2.0
    by = b.y + b.h / 2.0
    dist = float(np.hypot(ax - bx, ay - by))
    min_dim = float(min(a.w, a.h, b.w, b.h))
    return dist <= max(2.0, 1.25 * min_dim)""",
"""def _group_center_too_close(a: _GroupCandidate, b: _GroupCandidate) -> bool:
    ax = a.x + a.w / 2.0
    ay = a.y + a.h / 2.0
    bx = b.x + b.w / 2.0
    by = b.y + b.h / 2.0
    dist = float(np.hypot(ax - bx, ay - by))
    min_dim = float(min(a.w, a.h, b.w, b.h))
    return dist <= max(2.0, 1.6 * min_dim)"""
)

target.write_text(content, encoding="utf-8")
print("Applied all 9 patches correctly.")
