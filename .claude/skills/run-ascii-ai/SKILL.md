---
name: run-ascii-ai
description: Build, run, and drive the ascii_ai converter — turn an image or text into ASCII art. Use when asked to run/start ascii_ai, convert an image or a photo to ASCII, make ASCII art, render an image as text characters, glyph/braille/halfblock ASCII, build a FIGlet/text banner, produce colored/ANSI/HTML ASCII, or screenshot the ASCII output.
---

`ascii_ai` is a self-contained Python CLI that converts **an image** or **a
string** into ASCII art. Image modes: `brightness` (ramp), `edges` (DoG+Sobel
strokes), `combined`, `detail` (glyph/shape matching — highest structural
fidelity), `braille` (2×4 Unicode dots — 8× resolution), and `halfblock` (`▀`
fg/bg true-color — near-photographic). Plus 24-bit color, dithering
(floyd/atkinson/stucki), pre-processing (autocontrast/equalize/sharpen), and
text/ansi/html output. Drive it directly with `python3 ascii_ai.py ...`; the
committed smoke harness `.claude/skills/run-ascii-ai/driver.sh` exercises every
mode.

All paths below are relative to the repo root (the directory holding
`ascii_ai.py`).

## Prerequisites

Python 3.8+ and three pip packages. This is what actually worked in the
container (Python 3.11):

```bash
pip install -r requirements.txt
# equivalently: pip install Pillow numpy pyfiglet
```

No `apt-get` packages are needed — Pillow ships manylinux wheels. Chromium
(for the optional screenshot step below) is already present at
`/opt/pw-browsers/chromium-1194/chrome-linux/chrome`.

## Run (agent path) — the driver

Run the whole smoke suite (generates its own test image, checks every mode,
asserts non-empty output + exit codes, prints PASS/FAIL, exits non-zero on any
failure):

```bash
bash .claude/skills/run-ascii-ai/driver.sh
```

Expected tail:

```
passed: 15   failed: 0
ALL GREEN
```

`KEEP=1 bash .claude/skills/run-ascii-ai/driver.sh` keeps the scratch dir
(printed via `mktemp -d`) so you can inspect the generated `out.html`.

## Direct invocation — one mode at a time

You need a test image first. Any PNG/JPG works; generate one with:

```bash
python3 - <<'PY'
from PIL import Image, ImageDraw
img = Image.new("RGB", (320, 240), (60, 120, 230))
d = ImageDraw.Draw(img)
d.ellipse([230, 25, 300, 95], fill=(255, 230, 90))              # sun
d.polygon([(0, 240), (140, 120), (300, 240)], fill=(40, 120, 50))  # hill
img.save("sample.png")
PY
```

Image modes (dark→light ramps; `--width` is columns, height auto-corrected by
`--aspect 0.55`):

```bash
# 1. brightness ramp (default). Charsets: detailed(70)/standard/simple/blocks/custom
python3 ascii_ai.py image sample.png --width 70 --mode brightness --charset standard

# 2. structural edges — DoG + Sobel, angle→ | / - \  (raise --threshold for fewer edges)
python3 ascii_ai.py image sample.png --width 70 --mode edges --threshold 0.12

# 3. combined — edges drawn over the brightness fill
python3 ascii_ai.py image sample.png --width 70 --mode combined --charset simple

# 4. Floyd–Steinberg dithering with unicode blocks
python3 ascii_ai.py image sample.png --width 60 --charset blocks --dither

# 5. custom ramp + gamma + invert
python3 ascii_ai.py image sample.png --width 40 --charset custom --chars " .:oO@" --gamma 1.4 --invert

# 6. 24-bit ANSI color to the terminal
python3 ascii_ai.py image sample.png --width 60 --mode combined --color ansi --output ansi

# 7. self-contained HTML (colored) to a file
python3 ascii_ai.py image sample.png --width 120 --mode combined --color ansi --output html --out out.html
```

High-fidelity modes (best realism/detail — all verified this session):

```bash
# detail — glyph/shape matching against real font bitmaps (crisp structure).
# --cell HxW sets the match block (default 12x6); --font-file overrides the TTF.
python3 ascii_ai.py image sample.png --width 100 --mode detail --autocontrast

# braille — 2x4 Unicode dot cells, 8x spatial resolution; dither for gradients.
python3 ascii_ai.py image sample.png --width 80 --mode braille --dither --dither-mode atkinson

# halfblock — ▀ with fg=top / bg=bottom pixel; forces color, near-photographic.
python3 ascii_ai.py image sample.png --width 100 --mode halfblock --output html --out sphere.html

# preprocessing (any mode): stretch contrast, equalize, unsharp-mask sharpen.
python3 ascii_ai.py image sample.png --width 90 --autocontrast --equalize --sharpen 1.5
```

Text (FIGlet) mode:

```bash
python3 ascii_ai.py text "ASCII" --font slant          # banner
python3 ascii_ai.py text "OK" --color ansi --rgb 255,80,0
python3 ascii_ai.py text --list-fonts                   # 571 fonts available
```

Full option list: `python3 ascii_ai.py image -h` and `... text -h`.

## Screenshot the output (visual proof)

ANSI/HTML output is visual — render the HTML to PNG with the bundled Chromium
and look at it:

```bash
python3 ascii_ai.py image sample.png --width 120 --mode combined --color ansi --output html --out proof.html
/opt/pw-browsers/chromium-1194/chrome-linux/chrome --headless --no-sandbox \
  --disable-gpu --hide-scrollbars --screenshot="$PWD/proof.png" \
  --window-size=1200,780 "file://$PWD/proof.html"
```

`proof.png` shows the colored ASCII render (blue sky, yellow sun with a circular
edge outline, green hill). The `Failed to call method ... DBus` line chromium
prints is harmless — the PNG is still written.

## Gotchas

- **Ramps are ordered dark→light (index 0 = darkest).** A custom `--chars`
  string must follow that order or the image comes out as a negative. Add
  `--invert` to flip mapping without reordering.
- **Aspect correction is essential.** Terminal cells are ~2× taller than wide;
  without the default `--aspect 0.55` the picture looks vertically stretched.
  Lower it (e.g. `0.45`) if your terminal/font is unusually tall.
- **`--threshold` is on normalized magnitude (0..1), not 0..255.** Typical
  useful range is `0.10`–`0.25`; `edges` mode on a low-contrast image with a
  high threshold yields a nearly blank grid.
- **`--color ansi` only shows color with `--output ansi` (or `html`).** With
  `--output text` the escape codes are simply omitted; color is not lost, it's
  intentionally stripped for plain text.
- **Piping to `head` is safe.** The CLI swallows `BrokenPipeError`, so
  `... | head` won't dump a traceback.
- **`--charset detailed` (70 glyphs) needs width.** Below ~100 columns the extra
  tones blur into noise; use `standard`/`simple` for narrow renders.
- **The `text` subcommand needs a string OR `--list-fonts`.** Calling `text`
  with neither exits non-zero with a clear message.
- **`detail` mode needs a monospace TTF.** It auto-finds DejaVuSansMono /
  Liberation / FreeMono; if none exist it falls back to PIL's tiny bitmap font
  and quality drops. Pass `--font-file /path/to/Mono.ttf` to be explicit. It's
  the slowest mode but still sub-second at width 100 (renders ~95 glyph
  templates once, then one vectorized nearest-template search).
- **`braille` and `halfblock` need a capable viewer.** Braille glyphs require a
  font with U+2800 coverage; `halfblock` needs a **truecolor** terminal (or use
  `--output html` and screenshot, which always works). `halfblock` ignores
  `--color`/`--charset` — it's always `▀` with fg+bg color.
- **Black-background images fill with ink in mono modes.** `detail`/`braille`
  treat dark pixels as glyphs, so a subject on black comes out dense; add
  `--invert`, or use `halfblock` (color, so black stays black).
- **`--dither-mode atkinson`** gives punchier line art (braille), **`stucki`**
  smoother gradients, **`floyd`** (default) is the balanced middle.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'PIL'` | `pip install -r requirements.txt` (installs Pillow/numpy/pyfiglet). |
| `error: image not found: <path>` (exit 2) | Wrong path; the CLI checks and exits cleanly. Pass a real file. |
| `error: unknown font '<x>'` (exit 2) | Run `python3 ascii_ai.py text --list-fonts` and pick one. |
| Output looks like a photo negative | Your `--chars` ramp is light→dark; reverse it, or drop it and use `--invert`. |
| `edges` output is nearly blank | Lower `--threshold` (e.g. `0.10`) or `--dog 0` to skip the DoG pre-blur. |
| Chromium screenshot prints a `DBus` error | Harmless; the PNG is written anyway. |

## Test

There is no separate unit-test suite — `driver.sh` is the test. It is the
authoritative check that the tool works end-to-end.
