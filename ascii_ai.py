#!/usr/bin/env python3
"""
ascii_ai — convert an image or text into ASCII art.

Two subcommands:

    image  <path>   Convert a raster image into ASCII, with several rendering
                    modes (brightness ramp, structural edge detection, or both),
                    optional 24-bit ANSI color, dithering, and text/ansi/html
                    output.

    text   <string> Render a string as a large FIGlet-style banner (pyfiglet).

The image pipeline is NumPy-vectorized: the luminance grid is mapped to the
character ramp with a single indexed lookup rather than a per-pixel Python loop,
so even the 70-glyph "detailed" ramp on a wide render stays fast.

References that informed the algorithms:
  - Paul Bourke's ASCII grayscale ramps   (paulbourke.net/dataformats/asciiart/)
  - luminance = 0.299 R + 0.587 G + 0.114 B, ~0.55 vertical aspect correction
  - Difference-of-Gaussians + Sobel structural ASCII ("ASCII camera" approach):
    edge magnitude thresholds a cell, atan2(gy, gx) angle picks one of | / - \\
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


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_grid(path: str, width: int, aspect: float):
    """Load an image, resize to the ASCII grid, return (gray, rgb) float arrays.

    gray: (rows, cols) luminance in [0, 255]
    rgb:  (rows, cols, 3) uint8, aligned to the same grid (for color output)
    """
    from PIL import Image

    try:
        img = Image.open(path).convert("RGB")
    except FileNotFoundError:
        eprint(f"error: image not found: {path}")
        sys.exit(2)
    except Exception as exc:  # noqa: BLE001 - surface any decode error cleanly
        eprint(f"error: could not open image {path!r}: {exc}")
        sys.exit(2)

    w, h = img.size
    cols = max(1, width)
    # Each character cell is taller than wide; scale height by `aspect`.
    rows = max(1, int(round(cols * (h / w) * aspect)))
    img = img.resize((cols, rows), Image.LANCZOS)

    rgb = np.asarray(img, dtype=np.uint8)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.float32), rgb


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return gray
    norm = np.clip(gray / 255.0, 0.0, 1.0) ** (1.0 / gamma)
    return norm * 255.0


def floyd_steinberg(gray: np.ndarray, levels: int) -> np.ndarray:
    """Quantize `gray` (0..255) to `levels` bins with Floyd-Steinberg dithering.

    Returns an integer array of bin indices in [0, levels-1].
    """
    work = gray.astype(np.float32).copy()
    rows, cols = work.shape
    step = 255.0 / (levels - 1) if levels > 1 else 255.0
    for y in range(rows):
        for x in range(cols):
            old = work[y, x]
            new = round(old / step) * step
            work[y, x] = new
            err = old - new
            if x + 1 < cols:
                work[y, x + 1] += err * 7 / 16
            if y + 1 < rows:
                if x > 0:
                    work[y + 1, x - 1] += err * 3 / 16
                work[y + 1, x] += err * 5 / 16
                if x + 1 < cols:
                    work[y + 1, x + 1] += err * 1 / 16
    idx = np.clip(np.round(work / step), 0, levels - 1).astype(np.int32)
    return idx


def brightness_indices(gray: np.ndarray, ramp: str, invert: bool,
                       dither: bool) -> np.ndarray:
    """Map luminance to indices into `ramp` (0 == darkest glyph)."""
    n = len(ramp)
    if dither:
        idx = floyd_steinberg(gray, n)          # 0 == dark bin
    else:
        idx = np.clip((gray / 255.0) * (n - 1), 0, n - 1).astype(np.int32)
    if not invert:
        # Brighter pixels should use lighter (higher-index) glyphs already;
        # ramps are ordered dark->light so a bright pixel -> high index. Good.
        return idx
    return (n - 1) - idx


def sobel_edges(gray: np.ndarray, dog_sigma: float):
    """Difference-of-Gaussians preprocess, then Sobel. Return (mag, angle_idx).

    mag:       normalized edge magnitude in [0, 1]
    angle_idx: quantized orientation 0..3 indexing EDGE_CHARS
    """
    from PIL import Image, ImageFilter

    src = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    if dog_sigma > 0:
        g1 = np.asarray(src.filter(ImageFilter.GaussianBlur(dog_sigma)),
                        dtype=np.float32)
        g2 = np.asarray(src.filter(ImageFilter.GaussianBlur(dog_sigma * 1.6)),
                        dtype=np.float32)
        base = g1 - g2                      # difference of gaussians
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

    # Orientation of the edge line (perpendicular to gradient), 0..pi.
    theta = np.arctan2(gy, gx)              # -pi..pi (gradient direction)
    deg = (np.degrees(theta) + 180.0) % 180.0
    # Bucket into 4 orientations centered on 0/45/90/135.
    angle_idx = np.zeros_like(deg, dtype=np.int32)
    angle_idx[(deg >= 22.5) & (deg < 67.5)] = 3    # gradient ~45 -> edge "\"
    angle_idx[(deg >= 67.5) & (deg < 112.5)] = 0   # gradient ~90 -> edge "-"
    angle_idx[(deg >= 112.5) & (deg < 157.5)] = 1  # gradient ~135 -> edge "/"
    angle_idx[(deg >= 157.5) | (deg < 22.5)] = 2   # gradient ~0 -> edge "|"
    return mag, angle_idx


def _convolve3(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """3x3 convolution with edge padding, pure NumPy (no scipy)."""
    p = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
    return out


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------
def render_image(args) -> str:
    ramp = args.chars if args.charset == "custom" else RAMPS[args.charset]
    if args.charset == "custom" and not ramp:
        eprint("error: --charset custom requires --chars \"...\"")
        sys.exit(2)

    gray, rgb = load_grid(args.path, args.width, args.aspect)
    gray = apply_gamma(gray, args.gamma)

    rows, cols = gray.shape
    # Base layer: characters chosen per cell.
    if args.mode in ("brightness", "combined"):
        idx = brightness_indices(gray, ramp, args.invert, args.dither)
        char_grid = np.array(list(ramp))[idx]
    else:
        # edges-only: fill with spaces, edges drawn on top below.
        char_grid = np.full((rows, cols), " ", dtype="<U1")

    if args.mode in ("edges", "combined"):
        mag, angle_idx = sobel_edges(gray, args.dog)
        edge_mask = mag >= args.threshold
        edge_chars = np.array(EDGE_CHARS)[angle_idx]
        char_grid = np.where(edge_mask, edge_chars, char_grid)

    return _emit(char_grid, rgb, args)


def _emit(char_grid: np.ndarray, rgb: np.ndarray, args) -> str:
    rows, cols = char_grid.shape
    out_kind = args.output

    if out_kind == "text":
        return "\n".join("".join(char_grid[y]) for y in range(rows))

    if out_kind == "ansi":
        lines = []
        colored = args.color == "ansi"
        for y in range(rows):
            parts = []
            for x in range(cols):
                ch = char_grid[y, x]
                if colored:
                    r, g, b = (int(v) for v in rgb[y, x])
                    parts.append(f"\x1b[38;2;{r};{g};{b}m{ch}")
                else:
                    parts.append(ch)
            parts.append("\x1b[0m" if colored else "")
            lines.append("".join(parts))
        return "\n".join(lines)

    if out_kind == "html":
        esc = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}
        spans = []
        colored = args.color == "ansi"
        for y in range(rows):
            row_parts = []
            for x in range(cols):
                ch = char_grid[y, x]
                ch = esc.get(ch, ch)
                if colored:
                    r, g, b = (int(v) for v in rgb[y, x])
                    row_parts.append(
                        f'<span style="color:rgb({r},{g},{b})">{ch}</span>')
                else:
                    row_parts.append(ch)
            spans.append("".join(row_parts))
        body = "\n".join(spans)
        bg = "#000" if not args.invert else "#fff"
        fg = "#eee" if not args.invert else "#111"
        return (
            "<!doctype html><meta charset='utf-8'>"
            f"<pre style=\"font: 8px/8px monospace; background:{bg}; "
            f"color:{fg}; letter-spacing:0; padding:8px\">\n{body}\n</pre>"
        )

    raise ValueError(f"unknown output kind: {out_kind}")


def render_text(args) -> str:
    from pyfiglet import Figlet, FigletFont

    if args.list_fonts:
        return "\n".join(sorted(FigletFont.getFonts()))

    try:
        fig = Figlet(font=args.font, width=args.width)
    except Exception:  # pyfiglet raises FontNotFound
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ascii_ai",
        description="Convert an image or text into ASCII art.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # image ----------------------------------------------------------------
    pi = sub.add_parser("image", help="convert an image file to ASCII")
    pi.add_argument("path", help="path to the source image")
    pi.add_argument("--width", type=int, default=80,
                    help="output width in characters (default 80)")
    pi.add_argument("--aspect", type=float, default=0.55,
                    help="vertical aspect correction (default 0.55)")
    pi.add_argument("--mode", choices=["brightness", "edges", "combined"],
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
                    help="Floyd-Steinberg dithering before ramp mapping")
    pi.add_argument("--color", choices=["none", "ansi"], default="none")
    pi.add_argument("--output", choices=["text", "ansi", "html"],
                    default="text")
    pi.add_argument("--gamma", type=float, default=1.0,
                    help="gamma correction (>1 brightens midtones)")
    pi.add_argument("--threshold", type=float, default=0.15,
                    help="edge magnitude threshold 0..1 (edges/combined)")
    pi.add_argument("--dog", type=float, default=1.0,
                    help="difference-of-gaussians sigma (0 disables)")
    pi.add_argument("--out", default="",
                    help="write to file instead of stdout")
    pi.set_defaults(func=render_image)

    # text -----------------------------------------------------------------
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
            # Downstream pipe (e.g. `| head`) closed early; exit quietly.
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
