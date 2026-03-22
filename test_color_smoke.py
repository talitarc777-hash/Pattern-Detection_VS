import sys
import numpy as np
import cv2
from app.detector import DetectionConfig, _build_template_model, _detect_on_page

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
        
    if len(candidates) == 1 and abs(candidates[0].x - 90) < 5:
        print("COLOR SMOKE TEST PASSED")
    else:
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

    print("COLOR SMOKE TEST PASSED")

if __name__ == "__main__":
    run_test()
