"""Baseline accuracy test for sample training images."""
import sys
from pathlib import Path
from app.detector import DetectionConfig, detect_document, detect_document_multi, MarkupClass

SAMPLE_DIR = Path(__file__).parent / "sample"
OUT_DIR = Path(__file__).parent / "outputs" / "baseline_test"

def test_training1_markup1():
    """Expected ~81 detections."""
    result = detect_document(
        input_path=SAMPLE_DIR / "training1.jpeg",
        template_path=SAMPLE_DIR / "markup1.jpeg",
        output_dir=OUT_DIR / "t1_m1",
        config=DetectionConfig(),
    )
    print(f"training1 + markup1: {result.total_count} (expected ~81)")
    return result.total_count

def test_training1_markup1_2():
    """Expected ~38 detections."""
    result = detect_document(
        input_path=SAMPLE_DIR / "training1.jpeg",
        template_path=SAMPLE_DIR / "markup1_2.jpeg",
        output_dir=OUT_DIR / "t1_m1_2",
        config=DetectionConfig(),
    )
    print(f"training1 + markup1_2: {result.total_count} (expected ~38)")
    return result.total_count

def test_training2_circles():
    """Expected: brown=86, blue=3, green=2, black=36."""
    markups = [
        MarkupClass(name="brown", template_path=SAMPLE_DIR / "markupbrownCircle.jpeg"),
        MarkupClass(name="blue", template_path=SAMPLE_DIR / "markupBlueCircle.jpeg"),
        MarkupClass(name="green", template_path=SAMPLE_DIR / "markupGreenCircle.jpeg"),
        MarkupClass(name="black", template_path=SAMPLE_DIR / "markupBlackCircle.jpeg"),
    ]
    result = detect_document_multi(
        input_path=SAMPLE_DIR / "training2.jpg",
        markups=markups,
        output_dir=OUT_DIR / "t2_circles",
        config=DetectionConfig(),
    )
    print(f"training2 circles total: {result.total_count} (expected 127)")
    for name, count in result.class_totals:
        print(f"  {name}: {count}")
    print(f"  unclassified: {result.unclassified_count}")
    return result

if __name__ == "__main__":
    print("=== Baseline Accuracy Test ===")
    c1 = test_training1_markup1()
    c2 = test_training1_markup1_2()
    r3 = test_training2_circles()
    
    print("\n=== Summary ===")
    print(f"training1+markup1:   {c1}/81 = {c1/81*100:.1f}%")
    print(f"training1+markup1_2: {c2}/38 = {c2/38*100:.1f}%")
    totals = dict(r3.class_totals)
    print(f"training2 brown:     {totals.get('brown',0)}/86 = {totals.get('brown',0)/86*100:.1f}%")
    print(f"training2 blue:      {totals.get('blue',0)}/3 = {totals.get('blue',0)/3*100:.1f}%")
    print(f"training2 green:     {totals.get('green',0)}/2 = {totals.get('green',0)/2*100:.1f}%")
    print(f"training2 black:     {totals.get('black',0)}/36 = {totals.get('black',0)/36*100:.1f}%")
