# ascii-ai-skilll

Convert **images** or **text** into ASCII art from the command line, with
multiple rendering modes and configs.

## Install

```bash
pip install -r requirements.txt   # Pillow, numpy, pyfiglet
```

## Usage

```bash
# Image -> ASCII (brightness ramp, edges, or both)
python3 ascii_ai.py image photo.jpg --width 100 --mode combined --charset detailed
python3 ascii_ai.py image photo.jpg --width 100 --color ansi --output ansi   # colored terminal
python3 ascii_ai.py image photo.jpg --output html --out art.html             # self-contained HTML

# Text -> FIGlet banner
python3 ascii_ai.py text "HELLO" --font slant
python3 ascii_ai.py text --list-fonts
```

### Image options

| Flag | Purpose |
|---|---|
| `--width N` | Output width in characters (default 80) |
| `--aspect F` | Vertical aspect correction (default 0.55) |
| `--mode {brightness,edges,combined,detail,braille,halfblock}` | ramp · DoG+Sobel edges · both · glyph/shape matching · 2×4 braille dots · `▀` fg/bg color |
| `--dither-mode {floyd,atkinson,stucki}` | Error-diffusion kernel (with `--dither`) |
| `--autocontrast` / `--equalize` / `--sharpen F` | Pre-processing for extra detail |
| `--cell HxW` / `--font-file` | `detail`-mode match block and glyph font |
| `--charset {detailed,standard,simple,blocks,custom}` | Character ramp (`detailed` = Paul Bourke's 70-glyph) |
| `--chars "..."` | Custom ramp, dark→light (with `--charset custom`) |
| `--invert` | Flip brightness→glyph mapping |
| `--dither` | Floyd–Steinberg dithering |
| `--color {none,ansi}` | 24-bit ANSI truecolor (with `--output ansi`/`html`) |
| `--output {text,ansi,html}` | Output format |
| `--gamma F` / `--threshold F` / `--dog F` | Tone / edge-sensitivity / DoG-blur tuning |
| `--out FILE` | Write to a file instead of stdout |

Full help: `python3 ascii_ai.py image -h`, `python3 ascii_ai.py text -h`.

## Driving it as an agent

The repo ships a Claude Code skill at `.claude/skills/run-ascii-ai/`. Its
`driver.sh` is a smoke harness that generates a test image and exercises every
mode:

```bash
bash .claude/skills/run-ascii-ai/driver.sh   # -> "ALL GREEN"
```

## How it works

- **Brightness:** resize (with ~0.55 vertical aspect correction) → luminance
  `0.299R + 0.587G + 0.114B` → NumPy-indexed lookup into a dark→light character
  ramp.
- **Edges:** Difference-of-Gaussians preprocessing → Sobel gradients → magnitude
  thresholds a cell and `atan2(gy, gx)` orientation picks one of `| / - \`.
- **Detail:** each cell's sub-pixel block is matched to the best-fitting real
  font glyph (nearest-template search) — structure, not just density
  ([Xu, Zhang & Wong, *Structure-based ASCII Art*, 2010](https://ttwong12.github.io/papers/asciiart/asciiart.pdf)).
- **Braille:** 2×4 dots per cell (U+2800–U+28FF) → 8× spatial resolution, with
  Floyd/Atkinson/Stucki dithering.
- **Halfblock:** `▀` with foreground = top pixel and background = bottom pixel,
  doubling vertical resolution in true color (the chafa/viu approach).
- **Text:** [pyfiglet](https://pypi.org/project/pyfiglet/) FIGlet fonts.
