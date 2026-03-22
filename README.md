# Pattern Markup Counter

Counts repeated markup symbols in engineering/construction drawings (JPEG, PNG, PDF).
After detection, supports manual review to fix missed or incorrect identifications.

---

## Quick Start

1. Double-click `run.bat` — installs dependencies and launches the app
2. Pick your **input file** (drawing, scan, or PDF)
3. Pick your **markup template** (a cropped image of one symbol instance)
4. *(Optional)* click **Draw Scope** to restrict the detection region
5. Click **Run Detection** — results appear in the log and output folder
6. Click **Review Results** to add / remove detections manually

---

## User Workflow

```
Input file  ──►  Template image  ──►  [Scope]  ──►  Run Detection  ──►  Review Results
  (.pdf/.jpg/.png)  (one symbol crop)   (optional)    (auto-count)        (manual fix)
```

### 1 · Input File
- Raster: `.png`, `.jpg`, `.jpeg`
- Vector/mixed: `.pdf` (each page rendered and processed individually)

### 2 · Markup Template
Crop a **single clean instance** of the symbol you want to count and save it as a JPEG or PNG.
Rules for a good template:
- Best result: crop the markup directly from the same drawing/file you are detecting, if possible
- Tight crop — minimal blank margin around the symbol
- Same orientation as on the drawing (the detector is rotation-sensitive)
- Keep the original markup color and contrast (avoid converting to grayscale or using a recolored sample)

### 3 · Detection Scope (optional)
Restricts counting to a user-drawn polygon region.
- Click **Draw Scope** → hold left mouse and draw the boundary on the page preview
- Press `Enter` to confirm, `C` to clear the current drawing, `Esc` to cancel
- Detections outside the polygon are excluded from the count
- The scope is stored as normalized coordinates — it persists between runs

### 4 · Run Detection
The detector:
1. Builds a shape model from the template (mask, Hu moments, aspect ratio, fill ratio, LAB color signature)
2. Renders every page from the input file (at the chosen DPI)
3. Runs multi-view binarization (dark threshold, adaptive, background-distance, template-color)
4. **Runs `cv2.matchTemplate` sweeps** at multiple scales for pixel-level correlation
5. **Edge-based matching** (Canny) for line-art markups on dense drawings
6. Connected-component grouping scored against the template shape model
7. Merges all evidence sources, applies NMS, filters by scope
8. Saves annotated output images (numbered green boxes) into the output folder

Counts are shown in the **Run log** panel and in a summary popup.

### 5 · Review Results
After detection, click **Review Results** to open the review window.

| Key | Action |
|-----|--------|
| `A` | Switch to **add** mode — drag a box around a missed symbol |
| `D` | Switch to **delete** mode — click a wrong detection to remove it |
| `N` / `→` | Next page |
| `P` / `←` | Previous page |
| `S` | Save corrections and update counts |
| `Esc` | Cancel review without saving |

Corrected annotated images are saved back to the output folder.

---

## Detection Settings

| Setting | Default | Effect |
|---------|---------|--------|
| Match threshold | `0.45` | Combined detection confidence cutoff. Lower → more detections (risk: false positives). Higher → fewer (risk: misses). |
| PDF render DPI | `220` | Resolution for PDF rendering. Higher is slower but catches small symbols. Try `300` for very small marks. |
| Min scale | `0.18` | Smallest size of a symbol relative to the template. Reduce if on-page symbols are much smaller. |
| Max scale | `1.0` | Largest size relative to template. Increase if on-page symbols are larger. |
| Dark threshold | `0` (auto) | Pixel intensity cutoff for foreground extraction. Leave at 0 for auto-detection. |
| NMS IoU threshold | `0.35` | Overlap allowed between detections before suppression. Increase for densely packed symbols. |
| Num scales | `12` | How many scale steps to sweep during template matching. More = slower but more thorough. |

### Tuning for Too Many Detections (False Positives)
- Raise **Match threshold** (e.g. `0.45` → `0.55`)
- Lower **Max scale** if the template is much larger than on-page marks
- Use **Draw Scope** to exclude title blocks, legends, and notes

### Tuning for Missed Detections
- Lower **Match threshold** (e.g. `0.45` → `0.38`)
- Widen the **scale range** (lower Min scale, raise Max scale)
- For PDF files: raise **DPI** (e.g. `220` → `300`)
- If marks are faint/color: keep **Dark threshold** at `0` (auto)

---

## Output

Each run produces annotated images in the output folder:
```
outputs/
  drawing_page001_annotated.png   ← green numbered boxes for each detection
  drawing_page002_annotated.png
  ...
```
- **Green boxes** = detected markups
- **Green polygon** = scope boundary (if set, only marks inside are counted)
- The log panel shows total count and per-page breakdown

---

## CLI Mode (optional)

```powershell
python -m app.main --input "C:\path\to\drawing.pdf" --template "C:\path\to\markup.png" --output "C:\outputs"
```

Optional flags:
```
--threshold   0.45          match confidence cutoff
--dpi         220           PDF render DPI
--min-scale   0.18
--max-scale   1.0
--color-sensitivity auto    template color sensitivity: auto|soft|strict
--scope       poly:0.1,0.1;0.9,0.1;0.85,0.9;0.15,0.95
```

---

## Known Limitations

- **Rotation**: the detector assumes the markup appears at the same orientation as the template. Rotated variants may be missed.
- **Overlapping symbols**: heavily overlapping markups may be merged into one detection.
- **Multi-page PDFs**: each page is processed independently; scope applies to each page.
- **Very dense drawings**: title blocks, hatching, or dense annotation text can cause false positives. Use scope to exclude those areas.

---

## Detection Algorithm (Architecture)

```
Template image
   │
   ▼
_build_template_model()
   ├── foreground binary (Otsu + border-distance)
   ├── connected components → major parts
   ├── union mask, Hu moments, aspect/fill ratios
   └── LAB color signature (fg/bg mean, color_weight)

Page image  (per page)
   │
   ▼
Multi-view binarization ──►  component-grouping scoring ──►┐
       +                                                    ├──► NMS ──► scope filter ──► Candidates
cv2.matchTemplate sweep ─────────────────────────────────►┘
       +
 Edge-based matchTemplate ───────────────────────────────►┘
```

Scoring weights for component-group candidates:
- Mask IoU vs template: **34%**
- Dilated IoU: **18%**
- Hu moment similarity: **18%**
- Aspect ratio match: **12%**
- Layout similarity: **10%**
- Fill ratio match: **8%**
- Color similarity: blended in proportionally to template `color_weight`
