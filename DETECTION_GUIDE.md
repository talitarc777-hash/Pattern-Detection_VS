# Detection Guide

This note explains how the current detector works in plain language, what the settings mean, and how to improve detection quality.

It reflects the current implementation in `app/detector.py` and `app/main.py`.

## 1. What the detector is trying to do

The app is doing template-guided search.

You give it:

- an input page or PDF
- a template image of the markup you want to find

The detector then builds a model from the template and searches every page for regions that look similar enough in:

- shape
- size
- layout
- color, when the template color looks useful

## 2. How the template is understood

Before it looks at the page, the app turns the template into a small internal model.

From the template it extracts:

- the foreground mask
  - which pixels are treated as the actual markup, not background
- overall size and aspect ratio
- Hu moments
  - a compact shape signature
- component layout
  - if the template has 1, 2, or 3 main connected parts, the detector records their relative positions and sizes
- simple-blob shape traits for round filled marks
  - circularity
  - solidity
- color profile from foreground pixels only
  - LAB color statistics
  - hue, saturation, and chroma summary

If the template looks like a simple filled blob, such as a solid dot or circle, the detector uses a blob-friendly path. If it looks more complex, it also uses template matching and edge matching.

The template pre-processing also tries to avoid obvious contamination, such as a strong ring-like colored callout that does not look like the true symbol body.

## 3. How a page is searched

For each page, the detector creates several candidate maps instead of relying on only one method.

It builds binary or response maps from:

- dark-foreground thresholding
- color distance from page background
- adaptive thresholding
- CLAHE-enhanced thresholding
- template-color response, when the template color is strong enough to be useful

From those maps it extracts connected components and forms candidate groups.

It also runs extra search paths:

- simple blob search for filled circular marks
- multi-scale template matching
- multi-scale edge matching

This is why the detector can still find something even when one single method is weak on a particular page.

## 4. How the app decides a positive detection

A region becomes a final positive result only if it survives several filters.

### 4.1 Candidate-level checks

A candidate must first make sense structurally:

- its size must fall within the allowed scale range
- its connected parts must be compatible with the template layout
- its shape must be similar enough

For grouped component candidates, the score is mainly based on:

- mask overlap with the template
- dilated overlap
- Hu-moment similarity
- aspect-ratio similarity
- component-layout similarity

For simple blob candidates, the score is mainly based on:

- the dominant inner blob inside the candidate patch
- circularity
- aspect ratio
- solidity
- color similarity
- size consistency around the template mark

### 4.2 Color check

The detector does not use a hardcoded color like brown, red, or black.

Instead, it compares each candidate to the template's own foreground color profile.

That color profile is built from markup pixels only.

It does not score a candidate based on how strongly the markup contrasts with its local background.

If the template is clearly chromatic, color matters more.

If the template is close to grayscale, color matters less.

A candidate can be rejected if its color similarity is below the minimum required by the active color policy.

### 4.3 Match threshold

After shape and color are blended into a score, the candidate must meet the `match threshold`.

In simple terms:

- higher threshold = stricter, fewer results
- lower threshold = looser, more results

### 4.4 Duplicate removal

Many search paths may find the same markup.

The detector then runs NMS, which removes overlapping duplicate boxes and keeps the stronger one.

For simple filled marks, there is also a final blob-specific post-filter after NMS.

That post-filter re-checks:

- the refined inner blob, not just the raw connected component box
- markup-only color similarity
- blob-centeredness
- shape similarity
- effective size window

This extra step helps reject tiny text specks that have the right color but are still the wrong object.

### 4.5 Scope filter

If you draw a scope area, the final box must overlap the allowed region enough to survive.

So the real definition of a final positive result is:

1. it looks enough like the template in shape and scale
2. it passes the color check when color is active
3. its score is at or above the match threshold
4. it survives duplicate removal
5. it stays inside the chosen scope area, if scope is used

## 5. How color is handled now

The current detector uses template-relative color logic.

That means:

- it learns the color from the template you provide
- it does not assume a fixed markup color for the whole product
- it compares markup color to markup color, not markup-to-background contrast
- it becomes stricter when the template color is strong and distinctive
- it becomes softer when the template is gray, black, faded, or low-saturation

There is also a `color_sensitivity` control, exposed in the desktop UI as `Color matching` and in CLI as `--color-sensitivity`:

- `auto`
  - default
  - detector decides how much color should matter from the template
- `soft`
  - color matters less
  - useful if true marks vary a lot because of scan quality, fading, or compression
- `strict`
  - color matters more
  - useful when true marks in the file really keep the same color

## 6. Detection settings explained

## 6.1 Simple settings in the desktop app

These are the settings non-technical users should start with.

### Detection mode

This changes how strict the acceptance is.

| Mode | What it tries to do | Internal effect |
| --- | --- | --- |
| `Balanced (Recommended)` | Default starting point | Threshold `0.45`, min scale `0.18`, max scale `1.0`, NMS `0.35` |
| `Find More (Sensitive)` | Return more candidates, even weak ones | Threshold `0.38`, min scale `0.14`, max scale `1.05`, NMS `0.40` |
| `Avoid False Detections (Strict)` | Reject more weak lookalikes | Threshold `0.56`, min scale `0.20`, max scale `0.95`, NMS `0.30` |

Practical meaning:

- if the app misses true marks, try `Find More (Sensitive)`
- if the app finds too many wrong marks, try `Avoid False Detections (Strict)`

### Detail level

This changes how much resolution and scale search effort the detector uses.

| Level | What it tries to do | Internal effect |
| --- | --- | --- |
| `Faster` | Quicker run, less detail | DPI `180`, scale steps `10` |
| `Normal` | Best default | DPI `220`, scale steps `12` |
| `Tiny marks (Slower)` | Better for very small markup | DPI `300`, scale steps `16` |

Practical meaning:

- use `Tiny marks (Slower)` when markup is very small or faint
- use `Faster` only when speed matters more than recall

### Color matching

This controls how strongly the detector uses template color.

| UI value | Internal mode | What it tries to do |
| --- | --- | --- |
| `Auto (Recommended)` | `auto` | Decide automatically from the template whether color should matter a lot or only a little |
| `Allow More Variation` | `soft` | Be more tolerant when the same markup color shifts across pages because of scans, fading, or compression |
| `Same Color Only (Stricter)` | `strict` | Reject more wrong-color lookalikes when the real markup color is very consistent in the file |

Practical meaning:

- start with `Auto (Recommended)`
- switch to `Same Color Only (Stricter)` when true marks in the file really are the same color
- switch to `Allow More Variation` when true marks are the same kind of markup but their color is unstable in the scan

## 6.2 Advanced settings

These settings are available under `Show advanced settings`.

### Match threshold

This is the minimum final score needed for a detection.

- higher value
  - fewer detections
  - fewer false positives
- lower value
  - more detections
  - more false positives

Good rule:

- start near `0.45`
- go lower only when true marks are being missed
- go higher when obvious wrong matches are passing

### PDF render DPI

This controls how sharply PDF pages are rendered before detection.

- higher value
  - better for tiny marks and thin symbols
  - slower and heavier
- lower value
  - faster
  - can lose tiny details

Good rule:

- `220` is the normal default
- `300` is helpful for tiny markup

### Min scale

This is the smallest allowed size of a candidate relative to the template.

For simple filled marks, this now behaves more like a blob box-size check than a raw filled-area check.

- lower value
  - allows smaller candidates
  - helps when marks on the page are much smaller than the template
  - increases false positives
- higher value
  - ignores tiny candidates
  - reduces noise

### Max scale

This is the largest allowed size of a candidate relative to the template.

For simple filled marks, this also behaves more like a blob box-size check than a raw filled-area check.

- higher value
  - allows larger candidates
  - helps when marks on the page are much bigger than the template
  - increases false positives
- lower value
  - rejects oversized lookalikes

For tiny filled marks, the detector also applies a small internal upper-size buffer.

This is intentional, because rasterization and thresholding can make a real page dot measure slightly larger than the cropped template even when it is truly the same markup.

If the markup was cropped from the same file, the detector also uses a tighter lower-size bound to reject tiny specks.

### Dark threshold

This affects one of the page foreground masks.

`0` means auto mode, which is recommended first.

If you set it manually:

- higher value
  - treats lighter pixels as possible foreground
  - may help faint or light marks
  - can pull in more text and background noise
- lower value
  - keeps only darker pixels
  - may miss faint marks

### NMS IoU threshold

This controls duplicate suppression.

- lower value
  - merges nearby overlapping boxes more aggressively
  - reduces duplicate detections
- higher value
  - allows nearby boxes to survive more easily
  - may increase duplicates

### Num scale steps

This is how many sizes are tried during multi-scale matching.

- higher value
  - more chances to hit the correct size
  - slower
- lower value
  - faster
  - easier to miss the correct size

## 6.3 CLI form of the color setting

If you use command line mode, the same color control is available as:

### `--color-sensitivity auto|soft|strict`

Recommended use:

- `auto`
  - best default
- `soft`
  - use when the same markup color changes a lot across pages
- `strict`
  - use when markup color is very consistent and you want stronger rejection of wrong-color lookalikes

Example:

```powershell
python -m app.main --input "drawing.pdf" --template "mark.png" --output outputs --color-sensitivity strict
```

## 7. How to improve detection result

The improvements below usually help more than random threshold changes.

### 1. Use a template cropped from the same file when possible

This is one of the biggest wins.

Why it helps:

- same render style
- same line thickness
- same compression
- same scan noise
- same color profile

Best practice:

- crop one real markup from the same PDF or image set
- crop tightly around the symbol
- keep only a small amount of background

### 2. Keep the template clean

Avoid including:

- red circles
- arrows
- text labels
- nearby symbols
- long leader lines

The template should represent the actual markup, not the annotation around it.

### 3. Use consistent markup color to your advantage

Yes, if the markup color is truly consistent within the same file, that can improve detection accuracy.

Why:

- the detector already builds a foreground color profile from the template
- if true marks all share the same color, color becomes a strong discriminator
- wrong-color lookalikes become easier to reject

How to use this well:

- use a template cropped from the same file
- keep one run focused on one markup color
- in the UI, try `Same Color Only (Stricter)`
- in CLI, try `--color-sensitivity strict`

Important caution:

- if pages are faded, scanned differently, or compressed differently, `strict` can become too harsh
- in that case, `auto` is usually safer than `strict`

### 4. If one file contains multiple markup colors, do separate runs

If a PDF contains different colors for different meanings, do not expect one template to cover all of them well.

Better approach:

- one template per markup type or color
- one run per target symbol

This usually gives cleaner counts than trying to use one generic template for mixed colors.

### 5. Match the scale range to reality

If the true marks are always about the same size:

- raise `min scale`
- lower `max scale`

This removes many impossible candidates and improves precision.

If the detector is missing tiny marks:

- lower `min scale`
- use `Tiny marks (Slower)`
- increase DPI

### 6. Use scope to limit where the detector is allowed to search

If markups only appear in the drawing area, exclude:

- title blocks
- legends
- notes
- company logos
- revision tables

This reduces false positives a lot.

### 7. Change only one thing at a time

Good tuning order:

1. start with `Balanced + Normal`
2. improve the template crop
3. use scope if needed
4. switch to `Strict` or `Sensitive`
5. only then touch advanced values

### 8. Raise detail before lowering threshold too much

If you are missing tiny true marks, a better first move is often:

- higher DPI
- more scale steps
- tighter template crop

Lowering threshold too much often adds wrong matches faster than it recovers true ones.

## 8. Quick troubleshooting

### Too many false positives

Try this order:

1. use a tighter same-file template crop
2. use `Avoid False Detections (Strict)`
3. reduce `max scale` if oversized lookalikes are appearing
4. use scope
5. if color is stable, use `Same Color Only (Stricter)` in the UI or `--color-sensitivity strict` in CLI
6. raise `match threshold` slightly

### Too few detections

Try this order:

1. use a same-file template crop from a true positive
2. use `Find More (Sensitive)`
3. use `Tiny marks (Slower)`
4. lower `min scale` if the page marks are smaller than the template
5. if color varies a lot, stay on `Auto (Recommended)` or try `Allow More Variation`
6. lower `match threshold` slightly

### The template is a solid dot or circle

That case is supported, but it is also easier to confuse with other blobs.

For blob-like marks, accuracy improves when:

- color is consistent
- the crop is tight
- nearby text is excluded
- scale range is realistic

## 9. Direct answer to your color question

Yes, if the markup color on a map stays the same within that file, that can improve detection accuracy.

It helps because the detector already compares candidate color against the template's foreground color profile. When the file is color-consistent, the template color becomes a useful filter instead of just a weak hint.

The best practical setup is:

1. crop the template from the same file
2. keep the crop tight around the true markup
3. run one template per markup color or type
4. use `Same Color Only (Stricter)` in the UI, or `--color-sensitivity strict` in CLI

If the pages have large color variation from scan quality, keep `Auto (Recommended)` first and move to `Same Color Only (Stricter)` only after checking recall.
