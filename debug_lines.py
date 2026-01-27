#!/usr/bin/env python3
"""Quick debug harness for bordered line detection."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow running without installing the package
REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

# Default paths so you can run without CLI args
DEFAULT_IMAGE = "dark.png"
DEFAULT_OUT = "/tmp/lines_overlay.png"
DEFAULT_OUT_DIR = "/tmp"

from img2table.tables import threshold_dark_areas
from img2table.tables.metrics import compute_img_metrics
from img2table.tables.processing.bordered_tables.lines import detect_lines


def draw_lines(img, lines, color):
    for line in lines:
        cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), color, 1)

def detect_in_image(img_rgb):
    # Threshold + metrics
    thresh = threshold_dark_areas(img=img_rgb, char_length=11)
    char_length, median_line_sep, contours = compute_img_metrics(thresh.copy())

    if char_length is None:
        return None, None, None, None, None

    min_line_length = int(min(1.5 * median_line_sep, 4 * char_length)) if median_line_sep else 20

    h_lines, v_lines = detect_lines(
        img=img_rgb,
        contours=contours,
        char_length=char_length,
        min_line_length=min_line_length,
    )

    return h_lines, v_lines, char_length, median_line_sep, min_line_length


def save_overlay(img_bgr, h_lines, v_lines, out_path: Path) -> None:
    overlay = img_bgr.copy()
    # Draw horizontal in green, vertical in red
    draw_lines(overlay, h_lines, (0, 255, 0))
    draw_lines(overlay, v_lines, (0, 0, 255))
    cv2.imwrite(str(out_path), overlay)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug bordered table line detection.")
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output path for a single image")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for PDF pages")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 2

    if image_path.suffix.lower() == ".pdf":
        try:
            import pypdfium2
        except Exception as exc:  # pragma: no cover - runtime env specific
            print(f"pypdfium2 is required for PDF input: {exc}", file=sys.stderr)
            return 2

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = image_path.stem

        doc = pypdfium2.PdfDocument(str(image_path))
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                img_bgr = page.render(scale=200 / 72).to_numpy()
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                h_lines, v_lines, char_length, median_line_sep, min_line_length = detect_in_image(img_rgb)
                print(f"[page {page_idx + 1}] char_length={char_length} median_line_sep={median_line_sep} min_line_length={min_line_length}")
                if h_lines is None:
                    print(f"[page {page_idx + 1}] no text-like components detected")
                    continue

                print(f"[page {page_idx + 1}] h_lines={len(h_lines)} v_lines={len(v_lines)}")
                out_path = out_dir / f"{base}_page_{page_idx + 1}_lines.png"
                save_overlay(img_bgr, h_lines, v_lines, out_path)
                print(f"[page {page_idx + 1}] Saved overlay to {out_path}")
        finally:
            doc.close()
        return 0

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print("Failed to read image (cv2.imread returned None)", file=sys.stderr)
        return 2

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_lines, v_lines, char_length, median_line_sep, min_line_length = detect_in_image(img_rgb)

    if h_lines is None:
        print("Could not compute char_length; no text-like components detected.")
        return 1

    print(f"char_length={char_length:.2f} median_line_sep={median_line_sep} min_line_length={min_line_length}")
    print(f"h_lines={len(h_lines)} v_lines={len(v_lines)}")

    out_path = Path(args.out) if args.out else None
    if out_path:
        save_overlay(img_bgr, h_lines, v_lines, out_path)
        print(f"Saved overlay to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
