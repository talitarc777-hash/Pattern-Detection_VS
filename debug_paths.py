import sys
import numpy as np
import cv2
from app.detector import DetectionConfig, _build_template_model, _detect_on_page, _template_match_candidates, _edge_match_candidates, _score_candidate_groups

page = np.ones((400, 600, 3), dtype=np.uint8) * 240
cv2.rectangle(page, (90, 195), (110, 207), (0, 0, 255), -1)

tmpl = page[192:210, 87:114].copy()

cfg = DetectionConfig(match_threshold=0.40, min_scale=0.8, max_scale=1.2, num_scales=4, color_sensitivity="auto")
model = _build_template_model(tmpl)

print("Calling match_candidates...")
matches = _template_match_candidates(page, model.mask, model, cfg)
print("Template match candidates:", len(matches))

print("Calling edge_match_candidates...")
edges = _edge_match_candidates(page, model.mask, model, cfg)
print("Edge match candidates:", len(edges))

import app.detector
app.detector._detect_on_page = _detect_on_page # to be safe

# we can just print lengths of binaries
color_policy = app.detector._resolve_color_policy(model, cfg)
binaries = [
    app.detector._foreground_binary(page, dark_threshold=cfg.dark_threshold, for_template=False),
    app.detector._background_distance_binary(page),
    app.detector._adaptive_color_binary(page),
    app.detector._clahe_binary(page),
]
if color_policy.use_color_map:
    binaries.append(app.detector._template_color_binary(page, model, cfg))

total_groups = 0
for idx, binary in enumerate(binaries):
    comps = app.detector._filter_page_components(binary, model, cfg)
    groups = app.detector._score_candidate_groups(comps, page, model, cfg)
    print(f"Binary {idx} comps: {len(comps)}, groups: {len(groups)}")
    total_groups += len(groups)
    
print("Total groups:", total_groups)
