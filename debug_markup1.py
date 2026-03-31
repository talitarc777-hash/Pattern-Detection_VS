import sys
from pathlib import Path
from app.detector import DetectionConfig, detect_document

SAMPLE_DIR = Path(__file__).parent / "sample"
OUT_DIR = Path(__file__).parent / "outputs" / "markup1_debug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Running markup1 with debug tracing...")
cfg = DetectionConfig(debug_artifacts=True)

result = detect_document(
    input_path=SAMPLE_DIR / "training1.jpeg",
    template_path=SAMPLE_DIR / "markup1.jpeg",
    output_dir=OUT_DIR,
    config=cfg,
)

print(f"Total detected: {result.total_count}")
