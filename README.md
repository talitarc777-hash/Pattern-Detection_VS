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

The detector is tuned for noisy/blurred drawings by matching the template shape itself:
- supports connected symbols and split/multi-part symbols
- template-driven mask shape scoring instead of black-only assumptions
- adaptive contrast extraction and background-distance extraction for poor-quality images
- scale filtering + duplicate suppression (NMS)

If counts are too high (false positives):
- increase `Match threshold` (for example `0.50` to `0.65`)
- lower `Dark threshold` only if you are using manual value (for example `110` to `90`)

If counts are too low (missed detections):
- decrease `Match threshold` (for example `0.50` to `0.42`)
- increase `Dark threshold` only if needed (for example `90` to `115`) or keep `0` for auto
- widen scale range (increase `Max scale`)

## CLI mode (optional)

```powershell
python -m app.main --input "C:\path\to\1.jpeg" --template "C:\path\to\markup1.jpeg" --output "C:\path\to\outputs"
```
