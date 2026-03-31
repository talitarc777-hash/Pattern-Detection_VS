import time
import numpy as np
import cv2
from app.detector import DetectionConfig, _build_template_model, _detect_on_page, _post_filter_candidate

page = np.ones((400, 600, 3), dtype=np.uint8) * 240
cv2.rectangle(page, (90, 195), (110, 207), (0, 0, 255), -1) # Red (BGR)
cv2.rectangle(page, (290, 195), (310, 207), (255, 0, 0), -1) # Blue
cv2.rectangle(page, (490, 195), (510, 207), (30, 30, 30), -1) # Black

tmpl = page[192:210, 87:114].copy()

cfg = DetectionConfig(match_threshold=0.40, min_scale=0.8, max_scale=1.2, num_scales=4, color_sensitivity="auto")
model = _build_template_model(tmpl)

print(f"Area: {model.area}, W: {model.width}, H: {model.height}")
print(f"Circularity: {model.circularity}, Solidity: {model.solidity}, Aspect ratio: {model.aspect_ratio}")

# Let's monkey-patch _detect_on_page to print candidate lengths before NMS
import app.detector

original_nms = app.detector._nms_groups
def mocked_nms(candidates, iou):
    print(f"DEBUG: NMS called with {len(candidates)} candidates")
    if len(candidates) > 5000:
        print("DEBUG: EXTREME NMS, EXITING EARLY TO PREVENT HANG")
        import sys
        sys.exit(0)
    return original_nms(candidates, iou)

app.detector._nms_groups = mocked_nms

print("Testing detect_on_page...")
start = time.time()
cands = _detect_on_page(page, model, cfg)
print(f"Done in {time.time()-start:.2f}s, Kept: {len(cands)}")
