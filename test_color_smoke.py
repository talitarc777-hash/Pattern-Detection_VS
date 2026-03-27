import sys
import numpy as np
import cv2
from app.detector import DetectionConfig, _GroupCandidate, _build_template_model, _detect_on_page, _is_simple_blob_model, _nms_groups

def run_test():
    page = np.ones((400, 600, 3), dtype=np.uint8) * 240
    cv2.rectangle(page, (90, 195), (110, 207), (0, 0, 255), -1) # Red (BGR)
    cv2.rectangle(page, (290, 195), (310, 207), (255, 0, 0), -1) # Blue
    cv2.rectangle(page, (490, 195), (510, 207), (30, 30, 30), -1) # Black
    
    tmpl = page[192:210, 87:114].copy()
    
    cfg = DetectionConfig(match_threshold=0.40, min_scale=0.8, max_scale=1.2, num_scales=4, color_sensitivity="auto")
    model = _build_template_model(tmpl)
    
    print(f"Model color weight: {model.color_weight}")
    print(f"Template chromatic: {model.color_profile.chromatic}")
    
    candidates = _detect_on_page(page, model, cfg)
    print(f"Total candidates detected: {len(candidates)}")
    for c in candidates:
        print(f"Cand x={c.x} y={c.y} score={c.score}")

    if not any(abs(c.x - 90) < 18 and abs(c.y - 195) < 18 for c in candidates):
        print("COLOR SMOKE TEST FAILED")
        sys.exit(1)
    if any(abs(c.x - 290) < 25 and abs(c.y - 195) < 25 for c in candidates):
        print("COLOR SMOKE TEST FAILED")
        sys.exit(1)
    if any(abs(c.x - 490) < 25 and abs(c.y - 195) < 25 for c in candidates):
        print("COLOR SMOKE TEST FAILED")
        sys.exit(1)

    gray_page = np.ones((220, 420, 3), dtype=np.uint8) * 245
    cv2.rectangle(gray_page, (70, 92), (96, 112), (110, 110, 110), -1)
    cv2.rectangle(gray_page, (200, 92), (226, 112), (70, 70, 70), -1)
    cv2.rectangle(gray_page, (330, 92), (356, 112), (0, 0, 255), -1)

    gray_tmpl = gray_page[88:116, 66:100].copy()
    gray_cfg = DetectionConfig(match_threshold=0.38, min_scale=0.8, max_scale=1.2, num_scales=4, color_sensitivity="auto")
    gray_model = _build_template_model(gray_tmpl)
    gray_candidates = _detect_on_page(gray_page, gray_model, gray_cfg)

    print(f"Gray template color weight: {gray_model.color_weight}")
    print(f"Gray candidates: {len(gray_candidates)}")
    for c in gray_candidates:
        print(f"Gray cand x={c.x} y={c.y} score={c.score}")

    if not any(abs(c.x - 70) < 6 for c in gray_candidates):
        print("GRAY TEMPLATE SMOKE TEST FAILED")
        sys.exit(1)

    black_page = np.ones((260, 460, 3), dtype=np.uint8) * 247
    cv2.rectangle(black_page, (80, 95), (104, 119), (20, 20, 20), -1)
    cv2.rectangle(black_page, (260, 95), (284, 119), (35, 35, 35), -1)
    for x, y in ((170, 104), (184, 108), (198, 112), (214, 103), (228, 110), (340, 108), (354, 114), (368, 104)):
        cv2.rectangle(black_page, (x, y), (x + 5, y + 5), (20, 20, 20), -1)

    black_tmpl = black_page[90:124, 75:109].copy()
    black_cfg = DetectionConfig(match_threshold=0.38, min_scale=0.18, max_scale=1.0, num_scales=6, color_sensitivity="auto")
    black_model = _build_template_model(black_tmpl)
    black_candidates = _detect_on_page(black_page, black_model, black_cfg)

    print(f"Black template color weight: {black_model.color_weight}")
    print(f"Black template chromatic: {black_model.color_profile.chromatic}")
    print(f"Black candidates: {len(black_candidates)}")
    for c in black_candidates:
        print(f"Black cand x={c.x} y={c.y} score={c.score}")

    if not any(abs(c.x - 80) < 8 and abs(c.y - 95) < 8 for c in black_candidates):
        print("BLACK TEMPLATE SMOKE TEST FAILED: true dark blob not found")
        sys.exit(1)
    if any(abs(c.x - 170) < 16 and abs(c.y - 104) < 16 for c in black_candidates):
        print("BLACK TEMPLATE SMOKE TEST FAILED: tiny dark specks should be rejected")
        sys.exit(1)

    line_page = np.ones((220, 320, 3), dtype=np.uint8) * 252
    cv2.line(line_page, (80, 80), (80, 100), (0, 0, 0), 1)
    cv2.line(line_page, (86, 82), (86, 100), (0, 0, 0), 1)
    cv2.line(line_page, (80, 90), (86, 90), (0, 0, 0), 1)

    cv2.line(line_page, (170, 80), (170, 100), (0, 0, 0), 1)
    cv2.line(line_page, (176, 82), (176, 100), (0, 0, 0), 1)
    cv2.line(line_page, (240, 80), (240, 100), (0, 0, 0), 1)
    cv2.line(line_page, (246, 82), (246, 100), (0, 0, 0), 1)
    cv2.line(line_page, (240, 86), (246, 86), (0, 0, 0), 1)
    cv2.putText(line_page, "TP01", (140, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    line_tmpl = line_page[76:104, 76:90].copy()
    line_cfg = DetectionConfig(match_threshold=0.42, min_scale=0.8, max_scale=1.2, num_scales=6, color_sensitivity="auto")
    line_model = _build_template_model(line_tmpl)
    line_candidates = _detect_on_page(line_page, line_model, line_cfg)

    print(f"Line template simple blob model: {_is_simple_blob_model(line_model)}")
    print(f"Line template chromatic: {line_model.color_profile.chromatic}")
    print(f"Line template color weight: {line_model.color_weight}")
    print(f"Line candidates: {len(line_candidates)}")
    for c in line_candidates:
        print(f"Line cand x={c.x} y={c.y} score={c.score}")

    if _is_simple_blob_model(line_model):
        print("LINE TEMPLATE SMOKE TEST FAILED: misclassified as blob")
        sys.exit(1)
    if line_model.color_profile.chromatic or line_model.color_weight > 0.45:
        print("LINE TEMPLATE SMOKE TEST FAILED: near-black line treated as strongly chromatic")
        sys.exit(1)
    if not any(abs(c.x - 80) < 8 and abs(c.y - 80) < 8 for c in line_candidates):
        print("LINE TEMPLATE SMOKE TEST FAILED: true line target not found")
        sys.exit(1)

    nms_candidates = [
        _GroupCandidate(x=10, y=10, w=4, h=4, score=0.9, scale=1.0, component_ids=()),
        _GroupCandidate(x=40, y=40, w=4, h=4, score=0.8, scale=1.0, component_ids=()),
    ]
    nms_kept = _nms_groups(nms_candidates, 0.35)
    print(f"Empty-component NMS kept: {len(nms_kept)}")
    if len(nms_kept) != 2:
        print("NMS EMPTY-COMPONENT REGRESSION TEST FAILED")
        sys.exit(1)

    print("COLOR SMOKE TEST PASSED")

if __name__ == "__main__":
    run_test()
