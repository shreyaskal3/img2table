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
DEFAULT_IMAGE = "may-june (1)_page-0001.jpg"
DEFAULT_OUT = "./tmp/lines_overlay.png"
DEFAULT_OUT_DIR = "./tmp"

from img2table.tables import threshold_dark_areas
from img2table.tables.metrics import compute_img_metrics
from img2table.tables.processing.bordered_tables.lines import detect_lines
from img2table.tables.processing.borderless_tables import identify_borderless_tables


def draw_lines(img, lines, color):
    for line in lines:
        cv2.line(img, (line.x1, line.y1), (line.x2, line.y2), color, 1)


def detect_in_image(img_rgb, debug_base: Path | None = None):
    debug_threshold = debug_base / "threshold" if debug_base else None
    debug_metrics = debug_base / "metrics" if debug_base else None
    debug_lines = debug_base / "lines" if debug_base else None
    # Threshold + metrics
    thresh = threshold_dark_areas(img=img_rgb, char_length=11, debug_dir=debug_threshold)
    char_length, median_line_sep, contours = compute_img_metrics(thresh.copy(), debug_dir=debug_metrics)

    if char_length is None:
        return None, None, None, None, None, None, None

    min_line_length = int(min(1.5 * median_line_sep, 4 * char_length)) if median_line_sep else 20

    h_lines, v_lines = detect_lines(
        img=img_rgb,
        contours=contours,
        char_length=char_length,
        min_line_length=min_line_length,
        debug_dir=debug_lines,
    )

    return thresh, contours, h_lines, v_lines, char_length, median_line_sep, min_line_length


def save_overlay(img_bgr, h_lines, v_lines, out_path: Path) -> None:
    overlay = img_bgr.copy()
    # Draw horizontal in green, vertical in red
    draw_lines(overlay, h_lines, (0, 255, 0))
    draw_lines(overlay, v_lines, (0, 0, 255))
    cv2.imwrite(str(out_path), overlay)

def save_borderless_overlay(img_bgr, tables, out_path: Path) -> None:
    overlay = img_bgr.copy()
    for table in tables:
        cv2.rectangle(overlay, (table.x1, table.y1), (table.x2 - 1, table.y2 - 1), (255, 128, 0), 2)
        label = f"{table.nb_rows}x{table.nb_columns}"
        cv2.putText(overlay, label, (table.x1, max(0, table.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 128, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug table line detection (and optional borderless tables).")
    parser.add_argument("image", nargs="?", default=DEFAULT_IMAGE, help="Path to input image")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output path for a single image")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for PDF pages")
    parser.add_argument("--debug-dir", default="tmp", help="Base directory for debug images (optional)")
    parser.add_argument("--borderless", default=True ,action="store_true", help="Also run borderless table detection")
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
        debug_base = Path(args.debug_dir) if args.debug_dir else None

        doc = pypdfium2.PdfDocument(str(image_path))
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                img_bgr = page.render(scale=200 / 72).to_numpy()
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                thresh, contours, h_lines, v_lines, char_length, median_line_sep, min_line_length = detect_in_image(
                    img_rgb,
                    debug_base=(debug_base / f"page_{page_idx + 1}") if debug_base else None,
                )
                print(f"[page {page_idx + 1}] char_length={char_length} median_line_sep={median_line_sep} min_line_length={min_line_length}")
                if h_lines is None:
                    print(f"[page {page_idx + 1}] no text-like components detected")
                    continue

                print(f"[page {page_idx + 1}] h_lines={len(h_lines)} v_lines={len(v_lines)}")
                out_path = out_dir / f"{base}_page_{page_idx + 1}_lines.png"
                save_overlay(img_bgr, h_lines, v_lines, out_path)
                print(f"[page {page_idx + 1}] Saved overlay to {out_path}")

                if args.borderless:
                    if median_line_sep is None:
                        print(f"[page {page_idx + 1}] median_line_sep is None; skipping borderless detection")
                    else:
                        borderless_debug = (debug_base / f"page_{page_idx + 1}" / "borderless") if debug_base else None
                        borderless_tables = identify_borderless_tables(
                            thresh=thresh,
                            lines=h_lines + v_lines,
                            char_length=char_length,
                            median_line_sep=median_line_sep,
                            contours=contours,
                            existing_tables=[],
                            debug_dir=borderless_debug,
                        )
                        print(f"[page {page_idx + 1}] borderless_tables={len(borderless_tables)}")
                        out_path = out_dir / f"{base}_page_{page_idx + 1}_borderless.png"
                        save_borderless_overlay(img_bgr, borderless_tables, out_path)
                        print(f"[page {page_idx + 1}] Saved borderless overlay to {out_path}")
        finally:
            doc.close()
        return 0

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print("Failed to read image (cv2.imread returned None)", file=sys.stderr)
        return 2

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    debug_base = Path(args.debug_dir) if args.debug_dir else None
    thresh, contours, h_lines, v_lines, char_length, median_line_sep, min_line_length = detect_in_image(
        img_rgb,
        debug_base=debug_base,
    )

    if h_lines is None:
        print("Could not compute char_length; no text-like components detected.")
        return 1

    print(f"char_length={char_length:.2f} median_line_sep={median_line_sep} min_line_length={min_line_length}")
    print(f"h_lines={len(h_lines)} v_lines={len(v_lines)}")

    out_path = Path(args.out) if args.out else None
    if out_path:
        save_overlay(img_bgr, h_lines, v_lines, out_path)
        print(f"Saved overlay to {out_path}")

    if args.borderless:
        if median_line_sep is None:
            print("median_line_sep is None; skipping borderless detection")
        else:
            borderless_debug = (debug_base / "borderless") if debug_base else None
            borderless_tables = identify_borderless_tables(
                thresh=thresh,
                lines=h_lines + v_lines,
                char_length=char_length,
                median_line_sep=median_line_sep,
                contours=contours,
                existing_tables=[],
                debug_dir=borderless_debug,
            )
            print(f"borderless_tables={len(borderless_tables)}")
            if out_path:
                borderless_out = out_path.with_name(f"{out_path.stem}_borderless{out_path.suffix}")
            else:
                borderless_out = Path(DEFAULT_OUT).with_name("borderless_overlay.png")
            save_borderless_overlay(img_bgr, borderless_tables, borderless_out)
            print(f"Saved borderless overlay to {borderless_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
