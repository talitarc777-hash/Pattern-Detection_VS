import re
from pathlib import Path

target = Path("app/detector.py")
content = target.read_text(encoding="utf-8")

# Fix the variance check bug
content = content.replace(
"""    if float(np.std(template_l[model.mask > 0])) < 3.0:
        return []""",
"""    if float(np.std(template_l)) < 3.0:
        return []"""
)

target.write_text(content, encoding="utf-8")
print("Fixed variance check.")
