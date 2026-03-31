import re
from pathlib import Path

target = Path("app/detector.py")
content = target.read_text(encoding="utf-8")

old_text = """        if model.component_count > 1 and model.area <= 48:
            min_shape = max(min_shape, 0.30)
            min_part = 0.40
            min_gray = 0.42"""

new_text = """        if model.component_count > 1 and model.area <= 48:
            min_shape = max(min_shape, 0.18)
            min_part = 0.25
            min_gray = 0.30"""

if old_text not in content:
    print("WARNING: Text not found! The fix will silently fail.")
    
content = content.replace(old_text, new_text)

target.write_text(content, encoding="utf-8")
print("Successfully modified filter restrictions!")
