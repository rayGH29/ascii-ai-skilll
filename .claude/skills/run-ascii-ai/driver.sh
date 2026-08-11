#!/usr/bin/env bash
# driver.sh — smoke harness for the ascii_ai converter.
#
# Exercises every mode/config of the CLI and asserts each produces non-empty
# output and a zero exit code. Generates its own throwaway test image, so it
# needs no fixtures. Run from the repo root:
#
#     bash .claude/skills/run-ascii-ai/driver.sh
#
# Env:
#   PYTHON   python interpreter to use (default: python3)
#   KEEP=1   keep the scratch dir instead of deleting it on exit
set -uo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
APP="$ROOT/ascii_ai.py"
WORK="$(mktemp -d)"
IMG="$WORK/sample.png"
pass=0
fail=0

cleanup() { [ "${KEEP:-0}" = "1" ] || rm -rf "$WORK"; }
trap cleanup EXIT

if [ ! -f "$APP" ]; then
  echo "FATAL: cannot find ascii_ai.py at $APP" >&2
  exit 1
fi

# --- build a recognizable test image (sky + sun + hill + trunk) ------------
"$PYTHON" - "$IMG" <<'PY'
import sys
from PIL import Image, ImageDraw
W, H = 320, 240
img = Image.new("RGB", (W, H))
px = img.load()
for y in range(H):
    for x in range(W):
        px[x, y] = (int(60 + 120 * y / H), int(120 + 100 * y / H), 230)
d = ImageDraw.Draw(img)
d.ellipse([230, 25, 300, 95], fill=(255, 230, 90))
d.polygon([(0, 240), (140, 120), (300, 240)], fill=(40, 120, 50))
d.rectangle([120, 170, 175, 240], fill=(90, 60, 40))
img.save(sys.argv[1])
PY

# check($label, expected_min_bytes, cmd...) — runs cmd, asserts exit 0 and that
# stdout has at least the given number of bytes.
check() {
  local label="$1" minbytes="$2"; shift 2
  local out rc bytes
  out="$("$@" 2>"$WORK/err")"; rc=$?
  bytes=${#out}
  if [ $rc -eq 0 ] && [ "$bytes" -ge "$minbytes" ]; then
    printf 'PASS  %-34s (%d bytes)\n' "$label" "$bytes"
    pass=$((pass + 1))
  else
    printf 'FAIL  %-34s (rc=%d, %d bytes)\n' "$label" "$rc" "$bytes"
    sed 's/^/      /' "$WORK/err" >&2
    fail=$((fail + 1))
  fi
}

# expect_fail($label, cmd...) — asserts a NON-zero exit (graceful error path).
expect_fail() {
  local label="$1"; shift
  "$@" >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    printf 'PASS  %-34s (errored as expected)\n' "$label"
    pass=$((pass + 1))
  else
    printf 'FAIL  %-34s (should have errored)\n' "$label"
    fail=$((fail + 1))
  fi
}

echo "== image modes =="
check "brightness/standard"  200 "$PYTHON" "$APP" image "$IMG" --width 70 --mode brightness
check "brightness/detailed"  200 "$PYTHON" "$APP" image "$IMG" --width 70 --charset detailed --gamma 1.5
check "edges"                200 "$PYTHON" "$APP" image "$IMG" --width 70 --mode edges --threshold 0.12
check "combined"             200 "$PYTHON" "$APP" image "$IMG" --width 70 --mode combined --charset simple
check "dither/blocks"        200 "$PYTHON" "$APP" image "$IMG" --width 60 --charset blocks --dither
check "invert"               200 "$PYTHON" "$APP" image "$IMG" --width 60 --invert
check "custom charset"       200 "$PYTHON" "$APP" image "$IMG" --width 40 --charset custom --chars " .:oO@"
check "ansi color output"    500 "$PYTHON" "$APP" image "$IMG" --width 40 --color ansi --output ansi

echo "== upgraded modes =="
check "detail (glyph match)" 200 "$PYTHON" "$APP" image "$IMG" --width 70 --mode detail
check "detail + color"       500 "$PYTHON" "$APP" image "$IMG" --width 50 --mode detail --color ansi --output ansi
check "braille"              200 "$PYTHON" "$APP" image "$IMG" --width 60 --mode braille
check "braille + dither"     200 "$PYTHON" "$APP" image "$IMG" --width 60 --mode braille --dither --dither-mode atkinson
check "halfblock (color)"    500 "$PYTHON" "$APP" image "$IMG" --width 40 --mode halfblock --output ansi
check "dither stucki"        200 "$PYTHON" "$APP" image "$IMG" --width 60 --charset blocks --dither --dither-mode stucki
check "preprocess flags"     200 "$PYTHON" "$APP" image "$IMG" --width 60 --autocontrast --equalize --sharpen 1.5

echo "== html output (file) =="
"$PYTHON" "$APP" image "$IMG" --width 80 --mode combined --color ansi --output html --out "$WORK/out.html" 2>/dev/null
if [ -s "$WORK/out.html" ] && grep -q "<pre" "$WORK/out.html"; then
  echo "PASS  html file written                 ($(wc -c <"$WORK/out.html") bytes)"; pass=$((pass + 1))
else
  echo "FAIL  html file written"; fail=$((fail + 1))
fi

echo "== text mode =="
check "text banner (slant)"  100 "$PYTHON" "$APP" text "ASCII" --font slant
check "text color ansi"       50 "$PYTHON" "$APP" text "OK" --color ansi --rgb 255,80,0
check "list fonts"          1000 "$PYTHON" "$APP" text --list-fonts

echo "== error handling =="
expect_fail "missing image"  "$PYTHON" "$APP" image "$WORK/nope.png"
expect_fail "unknown font"   "$PYTHON" "$APP" text "hi" --font __nope__
expect_fail "empty text"     "$PYTHON" "$APP" text

echo
echo "----------------------------------------"
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ] && echo "ALL GREEN" || echo "FAILURES PRESENT"
exit "$fail"
