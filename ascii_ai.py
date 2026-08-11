#!/usr/bin/env python3
"""
ascii_ai — convert an image or text into ASCII art.

Two subcommands:

    image  <path>   Convert a raster image into ASCII. Rendering modes:
                      brightness  luminance -> character ramp (fast, classic)
                      edges       DoG + Sobel structural strokes ( | / - \\ )
                      combined    edges drawn over the brightness fill
                      detail      glyph/shape matching against real font
                                  bitmaps — highest structural fidelity
                      braille     2x4 Unicode braille dots — 8x resolution
                      halfblock   ▀ half-block color — 2x vertical, photographic
                    Plus 24-bit ANSI color, dithering (floyd/atkinson/stucki),
                    pre-processing (autocontrast/equalize/sharpen), and
                    text/ansi/html output.

    text   <string> Render a string as a large FIGlet-style banner (pyfiglet).

The image pipeline is NumPy-vectorized: ramp mapping is a single indexed
lookup, and glyph matching is a single nearest-template search
(cells · templatesᵀ), so even wide renders stay fast.

References that informed the algorithms:
  - Paul Bourke's ASCII grayscale ramps   (paulbourke.net/dataformats/asciiart/)
  - luminance = 0.299 R + 0.587 G + 0.114 B, ~0.55 vertical aspect correction
  - Difference-of-Gaussians + Sobel structural ASCII
  - Structure-based ASCII art via glyph/shape matching (Xu, Zhang, Wong, 2010)
  - Braille (U+2800..U+28FF) 2x4 dot cells; Floyd/Atkinson/Stucki dithering
  - Half-block ▀ fg/bg color rendering (chafa/viu approach)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Character ramps (dark -> light). Index 0 is the darkest glyph.
# ---------------------------------------------------------------------------
RAMPS = {
    # Paul Bourke's 70-level ramp, reversed so index 0 == darkest.
    "detailed": (
        "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    ),
    # A widely used medium ramp.
    "standard": "@%#*+=-:. ",
    # Bourke's classic short ramp (dark -> light).
    "simple": "@#*+:. ",
    # Unicode block shading, dark -> light.
    "blocks": "█▓▒░ ",
}

# Edge glyphs indexed by quantized gradient orientation.
#   0 -> horizontal edge, 1 -> diagonal /, 2 -> vertical, 3 -> diagonal \
EDGE_CHARS = ["-", "/", "|", "\\"]

# Error-diffusion dithering kernels: (divisor, [(dx, dy, weight), ...]).
DITHER_KERNELS = {
    "floyd": (16, [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]),
    "atkinson": (8, [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1),
                     (1, 1, 1), (0, 2, 1)]),
    "stucki": (42, [(1, 0, 8), (2, 0, 4),
                    (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                    (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)]),
}

# Default candidate glyph set for `detail` mode (printable ASCII, space first).
DETAIL_GLYPHS = (
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


# ---------------------------------------------------------------------------
# Image loading + preprocessing
# ---------------------------------------------------------------------------
def open_image(path, autocontrast=False, equalize=False, sharpen=0.0):
    """Open an RGB image and apply optional detail-boosting preprocessing."""
    from PIL import Image, ImageOps, ImageFilter

    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        eprint(f"error: image not found: {path}")
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 - surface any decode error cleanly
        eprint(f"error: could not open image {path!r}: {exc}")
        sys.exit(2)

    if sharpen and sharpen > 0:
        img = img.filter(ImageFilter.UnsharpMask(
            radius=2, percent=int(sharpen * 100), threshold=2))
    if equalize:
        img = ImageOps.equalize(img)
    if autocontrast:
        img = ImageOps.autocontrast(img, cutoff=1)
    return img


def grid_dims(img, width, aspect):
    w, h = img.size
    cols = max(1, width)
    rows = max(1, int(round(cols * (h / w) * aspect)))
    return cols, rows


def sample(img, cols, rows):
    """Resize to (cols, rows) and return (gray, rgb) aligned arrays."""
    from PIL import Image

    im = img.resize((cols, rows), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.uint8)
    return _luminance(rgb), rgb


def load_grid(path, width, aspect, autocontrast=False, equalize=False,
              sharpen=0.0):
    """Back-compat helper: preprocess, size the char grid, sample once."""
    img = open_image(path, autocontrast, equalize, sharpen)
    cols, rows = grid_dims(img, width, aspect)
    return sample(img, cols, rows)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return gray
    norm = np.clip(gray / 255.0, 0.0, 1.0) ** (1.0 / gamma)
    return norm * 255.0


# ---------------------------------------------------------------------------
# Dithering
# ---------------------------------------------------------------------------
def error_diffuse(gray: np.ndarray, levels: int, mode: str) -> np.ndarray:
    """Quantize `gray` (0..255) to `levels` bins with error-diffusion dithering.

    Returns integer bin indices in [0, levels-1]. `mode` selects the kernel
    (floyd / atkinson / stucki).
    """
    div, kern = DITHER_KERNELS[mode]
    work = gray.astype(np.float32).copy()
    rows, cols = work.shape
    step = 255.0 / (levels - 1) if levels > 1 else 255.0
    for y in range(rows):
        for x in range(cols):
            old = work[y, x]
            new = round(old / step) * step
            work[y, x] = new
            err = old - new
            for dx, dy, w in kern:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows:
                    work[ny, nx] += err * w / div
    return np.clip(np.round(work / step), 0, levels - 1).astype(np.int32)


def floyd_steinberg(gray: np.ndarray, levels: int) -> np.ndarray:
    """Back-compat wrapper: Floyd-Steinberg quantization to `levels` bins."""
    return error_diffuse(gray, levels, "floyd")


def brightness_indices(gray, ramp, invert, dither, dither_mode="floyd"):
    """Map luminance to indices into `ramp` (0 == darkest glyph)."""
    n = len(ramp)
    if dither:
        idx = error_diffuse(gray, n, dither_mode)   # 0 == dark bin
    else:
        idx = np.clip((gray / 255.0) * (n - 1), 0, n - 1).astype(np.int32)
    if not invert:
        return idx
    return (n - 1) - idx


# ---------------------------------------------------------------------------
# Structural edges (DoG + Sobel)
# ---------------------------------------------------------------------------
def sobel_edges(gray: np.ndarray, dog_sigma: float):
    from PIL import Image, ImageFilter

    src = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    if dog_sigma > 0:
        g1 = np.asarray(src.filter(ImageFilter.GaussianBlur(dog_sigma)),
                        dtype=np.float32)
        g2 = np.asarray(src.filter(ImageFilter.GaussianBlur(dog_sigma * 1.6)),
                        dtype=np.float32)
        base = g1 - g2
    else:
        base = gray.astype(np.float32)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = _convolve3(base, kx)
    gy = _convolve3(base, ky)

    mag = np.hypot(gx, gy)
    mmax = mag.max()
    if mmax > 0:
        mag = mag / mmax

    theta = np.arctan2(gy, gx)
    deg = (np.degrees(theta) + 180.0) % 180.0
    angle_idx = np.zeros_like(deg, dtype=np.int32)
    angle_idx[(deg >= 22.5) & (deg < 67.5)] = 3
    angle_idx[(deg >= 67.5) & (deg < 112.5)] = 0
    angle_idx[(deg >= 112.5) & (deg < 157.5)] = 1
    angle_idx[(deg >= 157.5) | (deg < 22.5)] = 2
    return mag, angle_idx


def _convolve3(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out


# ---------------------------------------------------------------------------
# detail mode — glyph/shape matching against real font bitmaps
# ---------------------------------------------------------------------------
def _resolve_font(font_file):
    from PIL import ImageFont

    candidates = [font_file] if font_file else []
    candidates += FONT_CANDIDATES
    for path in candidates:
        if not path:
            continue
        try:
            return path, ImageFont.truetype(path, 16)  # size set later
        except Exception:
            continue
    return None, ImageFont.load_default()


def build_templates(glyphs, ch, cw, font_file):
    """Render each glyph to a (ch*cw) ink-coverage vector in [0,1]."""
    from PIL import Image, ImageDraw, ImageFont

    path, _ = _resolve_font(font_file)
    if path:
        font = ImageFont.truetype(path, ch)
    else:
        font = ImageFont.load_default()

    tpls = []
    for gch in glyphs:
        im = Image.new("L", (cw, ch), 0)          # black background
        d = ImageDraw.Draw(im)
        try:
            bbox = d.textbbox((0, 0), gch, font=font)
        except Exception:
            bbox = (0, 0, cw, ch)
        gw = bbox[2] - bbox[0]
        gh = bbox[3] - bbox[1]
        x = (cw - gw) / 2 - bbox[0]
        y = (ch - gh) / 2 - bbox[1]
        d.text((x, y), gch, fill=255, font=font)   # white ink
        tpls.append((np.asarray(im, dtype=np.float32) / 255.0).reshape(-1))
    return np.stack(tpls)                           # (G, ch*cw)


def render_detail(img, cols, rows, args):
    """Match every cell's sub-pixel block to the best-fitting glyph."""
    ch, cw = args.cell                              # (height, width) in px
    glyphs = args.glyphs or DETAIL_GLYPHS
    templates = build_templates(glyphs, ch, cw, args.font_file)   # (G, D)

    gray, _ = sample(img, cols * cw, rows * ch)     # hi-res luminance
    gray = apply_gamma(gray, args.gamma)
    dark = 1.0 - np.clip(gray / 255.0, 0, 1)        # ink = darkness
    if args.invert:
        dark = 1.0 - dark

    # (rows, ch, cols, cw) -> (rows*cols, ch*cw)
    blocks = (dark.reshape(rows, ch, cols, cw)
                  .transpose(0, 2, 1, 3)
                  .reshape(rows * cols, ch * cw))
    # argmin_g ||block - tpl||^2 ; drop the constant ||block||^2 term.
    tpl2 = (templates ** 2).sum(axis=1)             # (G,)
    dist = tpl2[None, :] - 2.0 * (blocks @ templates.T)
    idx = dist.argmin(axis=1)
    char_grid = np.array(list(glyphs))[idx].reshape(rows, cols)

    fg = None
    if args.color == "ansi":
        _, rgb = sample(img, cols, rows)
        fg = rgb
    return char_grid, fg


# ---------------------------------------------------------------------------
# braille mode — 2x4 Unicode dot cells (U+2800..U+28FF)
# ---------------------------------------------------------------------------
# Dot bit for (col in {0,1}, row in {0,1,2,3}) per the Unicode braille layout.
_BRAILLE_BIT = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}


def render_braille(img, cols, rows, args):
    gray, _ = sample(img, cols * 2, rows * 4)       # dot resolution
    gray = apply_gamma(gray, args.gamma)
    dark = gray if not args.invert else (255.0 - gray)

    if args.dither:
        # index 0 == darkest bin == dot ON
        on = (error_diffuse(dark, 2, args.dither_mode) == 0)
    else:
        on = dark < (args.threshold * 255.0)

    masks = np.zeros((rows, cols), dtype=np.int32)
    for (c, r), bit in _BRAILLE_BIT.items():
        masks += bit * on[r::4, c::2][:rows, :cols].astype(np.int32)
    char_grid = np.array([chr(0x2800 + m) for m in masks.ravel()],
                         dtype="<U1").reshape(rows, cols)

    fg = None
    if args.color == "ansi":
        _, rgb = sample(img, cols, rows)
        fg = rgb
    return char_grid, fg


# ---------------------------------------------------------------------------
# halfblock mode — ▀ with fg=top pixel, bg=bottom pixel (2x vertical, color)
# ---------------------------------------------------------------------------
def render_halfblock(img, cols, rows, args):
    _, rgb = sample(img, cols, rows * 2)            # two stacked pixels/char
    top = rgb[0::2][:rows]
    bot = rgb[1::2][:rows]
    char_grid = np.full((rows, cols), "▀", dtype="<U1")
    return char_grid, top, bot


# ---------------------------------------------------------------------------
# Renderers / emit
# ---------------------------------------------------------------------------
def render_image(args) -> str:
    img = open_image(args.path, args.autocontrast, args.equalize, args.sharpen)
    cols, rows = grid_dims(img, args.width, args.aspect)

    if args.mode == "detail":
        char_grid, fg = render_detail(img, cols, rows, args)
        return _emit(char_grid, args, fg=fg)

    if args.mode == "braille":
        char_grid, fg = render_braille(img, cols, rows, args)
        return _emit(char_grid, args, fg=fg)

    if args.mode == "halfblock":
        char_grid, fg, bg = render_halfblock(img, cols, rows, args)
        return _emit(char_grid, args, fg=fg, bg=bg)

    # ramp-based modes (brightness / edges / combined)
    ramp = args.chars if args.charset == "custom" else RAMPS[args.charset]
    if args.charset == "custom" and not ramp:
        eprint("error: --charset custom requires --chars \"...\"")
        sys.exit(2)

    gray, rgb = sample(img, cols, rows)
    gray = apply_gamma(gray, args.gamma)

    if args.mode in ("brightness", "combined"):
        idx = brightness_indices(gray, ramp, args.invert, args.dither,
                                 args.dither_mode)
        char_grid = np.array(list(ramp))[idx]
    else:
        char_grid = np.full((rows, cols), " ", dtype="<U1")

    if args.mode in ("edges", "combined"):
        mag, angle_idx = sobel_edges(gray, args.dog)
        edge_mask = mag >= args.threshold
        char_grid = np.where(edge_mask, np.array(EDGE_CHARS)[angle_idx],
                             char_grid)

    fg = rgb if args.color == "ansi" else None
    return _emit(char_grid, args, fg=fg)


def _emit(char_grid, args, fg=None, bg=None) -> str:
    """Render the char grid to text / ansi / html.

    fg: optional (rows, cols, 3) foreground colors.
    bg: optional (rows, cols, 3) background colors (halfblock).
    """
    rows, cols = char_grid.shape
    out_kind = args.output

    if out_kind == "text":
        return "\n".join("".join(char_grid[y]) for y in range(rows))

    if out_kind == "ansi":
        lines = []
        for y in range(rows):
            parts = []
            for x in range(cols):
                pre = ""
                if fg is not None:
                    r, g, b = (int(v) for v in fg[y, x])
                    pre += f"\x1b[38;2;{r};{g};{b}m"
                if bg is not None:
                    r, g, b = (int(v) for v in bg[y, x])
                    pre += f"\x1b[48;2;{r};{g};{b}m"
                parts.append(pre + char_grid[y, x])
            lines.append("".join(parts) + ("\x1b[0m" if (fg is not None
                                                         or bg is not None)
                                           else ""))
        return "\n".join(lines)

    if out_kind == "html":
        esc = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
        spans = []
        for y in range(rows):
            row_parts = []
            for x in range(cols):
                ch = esc.get(char_grid[y, x], char_grid[y, x])
                styles = []
                if fg is not None:
                    r, g, b = (int(v) for v in fg[y, x])
                    styles.append(f"color:rgb({r},{g},{b})")
                if bg is not None:
                    r, g, b = (int(v) for v in bg[y, x])
                    styles.append(f"background:rgb({r},{g},{b})")
                if styles:
                    row_parts.append(
                        f'<span style="{";".join(styles)}">{ch}</span>')
                else:
                    row_parts.append(ch)
            spans.append("".join(row_parts))
        body = "\n".join(spans)
        page_bg = "#000" if not args.invert else "#fff"
        page_fg = "#eee" if not args.invert else "#111"
        return (
            "<!doctype html><meta charset='utf-8'>"
            f"<pre style=\"font: 8px/8px monospace; background:{page_bg}; "
            f"color:{page_fg}; letter-spacing:0; line-height:1; "
            f"padding:8px\">\n{body}\n</pre>"
        )

    raise ValueError(f"unknown output kind: {out_kind}")


def render_text(args) -> str:
    from pyfiglet import Figlet, FigletFont

    if args.list_fonts:
        return "\n".join(sorted(FigletFont.getFonts()))

    try:
        fig = Figlet(font=args.font, width=args.width)
    except Exception:
        eprint(f"error: unknown font {args.font!r}. "
               f"Try `text --list-fonts` to see options.")
        sys.exit(2)

    banner = fig.renderText(args.text).rstrip("\n")
    if args.color == "ansi":
        banner = f"\x1b[38;2;{args.rgb[0]};{args.rgb[1]};{args.rgb[2]}m" \
                 f"{banner}\x1b[0m"
    return banner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _rgb_triple(s: str):
    try:
        parts = [int(p) for p in s.split(",")]
        assert len(parts) == 3 and all(0 <= p <= 255 for p in parts)
        return tuple(parts)
    except Exception:
        raise argparse.ArgumentTypeError("expected R,G,B with each 0-255")


def _cell(s: str):
    try:
        h, w = (int(p) for p in s.lower().split("x"))
        assert h > 0 and w > 0
        return (h, w)
    except Exception:
        raise argparse.ArgumentTypeError("expected HxW, e.g. 12x6")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ascii_ai",
        description="Convert an image or text into ASCII art.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("image", help="convert an image file to ASCII")
    pi.add_argument("path", help="path to the source image")
    pi.add_argument("--width", type=int, default=80,
                    help="output width in characters (default 80)")
    pi.add_argument("--aspect", type=float, default=0.55,
                    help="vertical aspect correction (default 0.55)")
    pi.add_argument("--mode",
                    choices=["brightness", "edges", "combined", "detail",
                             "braille", "halfblock"],
                    default="brightness")
    pi.add_argument("--charset",
                    choices=["detailed", "standard", "simple", "blocks",
                             "custom"],
                    default="standard")
    pi.add_argument("--chars", default="",
                    help="custom ramp (dark->light) when --charset custom")
    pi.add_argument("--invert", action="store_true",
                    help="invert brightness->glyph mapping")
    pi.add_argument("--dither", action="store_true",
                    help="error-diffusion dithering (ramp & braille modes)")
    pi.add_argument("--dither-mode", choices=["floyd", "atkinson", "stucki"],
                    default="floyd", help="dithering kernel (default floyd)")
    pi.add_argument("--color", choices=["none", "ansi"], default="none")
    pi.add_argument("--output", choices=["text", "ansi", "html"],
                    default="text")
    pi.add_argument("--gamma", type=float, default=1.0,
                    help="gamma correction (>1 brightens midtones)")
    pi.add_argument("--threshold", type=float, default=0.15,
                    help="edge/braille threshold 0..1")
    pi.add_argument("--dog", type=float, default=1.0,
                    help="difference-of-gaussians sigma (0 disables)")
    # preprocessing (all modes)
    pi.add_argument("--autocontrast", action="store_true",
                    help="stretch contrast before converting")
    pi.add_argument("--equalize", action="store_true",
                    help="histogram-equalize before converting")
    pi.add_argument("--sharpen", type=float, default=0.0,
                    help="unsharp-mask amount, e.g. 1.5 (0 disables)")
    # detail mode
    pi.add_argument("--cell", type=_cell, default=(12, 6),
                    help="detail cell size HxW in px (default 12x6)")
    pi.add_argument("--glyphs", default="",
                    help="detail candidate glyphs (default: built-in set)")
    pi.add_argument("--font-file", default="",
                    help="TTF used to render detail glyphs")
    pi.add_argument("--out", default="",
                    help="write to file instead of stdout")
    pi.set_defaults(func=render_image)

    pt = sub.add_parser("text", help="render a string as a FIGlet banner")
    pt.add_argument("text", nargs="?", default="", help="text to render")
    pt.add_argument("--font", default="standard")
    pt.add_argument("--width", type=int, default=80)
    pt.add_argument("--color", choices=["none", "ansi"], default="none")
    pt.add_argument("--rgb", type=_rgb_triple, default=(0, 255, 150),
                    help="ANSI color as R,G,B (with --color ansi)")
    pt.add_argument("--list-fonts", action="store_true",
                    help="list available fonts and exit")
    pt.add_argument("--out", default="")
    pt.set_defaults(func=render_text)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "text" and not args.list_fonts and not args.text:
        eprint("error: text command requires a string (or --list-fonts)")
        return 2
    result = args.func(args)
    if getattr(args, "out", ""):
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result + "\n")
        eprint(f"wrote {args.out}")
    else:
        try:
            print(result)
        except BrokenPipeError:
            try:
                sys.stdout.close()
            except Exception:
                pass
            return 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
