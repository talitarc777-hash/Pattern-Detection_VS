# Pattern Markup Counter (Windows)

This application counts repeated markup symbols in drawing/image files.

Supported input:
- Raster: `.png`, `.jpg`, `.jpeg`
- Vector/mixed: `.pdf` (rendered page-by-page)

## One-click run

1. Double-click `run.bat`
2. In the app:
- choose input file (for example `1.jpeg` or a `.pdf`)
- choose template symbol image (for example `markup1.jpeg`)
- choose output folder
- click **Run Detection**

The app outputs:
- total markup count
- per-page count
- annotated images in output folder

## Notes on low-quality files

The detector is tuned for noisy/blurred drawings by matching the black symbol geometry from the template:
- left straight stroke + right reversed-L stroke
- connected-component pairing with template-shape scoring
- scale filtering + duplicate suppression (NMS)

If counts are too high (false positives):
- increase `Match threshold` (for example `0.60` to `0.72`)
- lower `Dark threshold` (for example `90` to `75`)

If counts are too low (missed detections):
- decrease `Match threshold` (for example `0.60` to `0.52`)
- increase `Dark threshold` (for example `90` to `105`)
- widen scale range (increase `Max scale`)

## CLI mode (optional)

```powershell
python -m app.main --input "C:\path\to\1.jpeg" --template "C:\path\to\markup1.jpeg" --output "C:\path\to\outputs"
```
