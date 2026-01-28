from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell
from img2table.tables.objects.line import Line


def identify_straight_lines(
    thresh: np.ndarray,
    min_line_length: float,
    char_length: float,
    vertical: bool = True,
    debug_dir: Optional[str | Path] = None,
    debug_prefix: str = "",
) -> list[Line]:
    """
    Identify straight lines in image in a specific direction
    :param thresh: thresholded edge image
    :param min_line_length: minimum line length
    :param char_length: average character length
    :param vertical: boolean indicating if vertical lines are detected
    :param debug_dir: optional directory to dump debug images
    :param debug_prefix: prefix for debug images
    :return: list of detected lines
    """
    debug_dir_path = Path(debug_dir) if debug_dir is not None else None
    if debug_dir_path is not None:
        debug_dir_path.mkdir(parents=True, exist_ok=True)

    def write_debug(name: str, image: np.ndarray) -> None:
        if debug_dir_path is None:
            return
        if image.dtype == np.bool_:
            image = image.astype(np.uint8) * 255
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        prefix = f"{debug_prefix}_" if debug_prefix else ""
        cv2.imwrite(str(debug_dir_path / f"{prefix}{name}.png"), image)

    write_debug("binary_input", thresh)

    # Apply masking on image
    kernel_dims = (1, round(min_line_length / 3) or 1) if vertical else (round(min_line_length / 3) or 1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    write_debug("mask_open", mask)

    # Apply closing for hollow lines
    hollow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1) if vertical else (1, 3))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hollow_kernel)
    write_debug("mask_closed", mask_closed)

    # Apply closing for dotted lines
    dotted_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, round(min_line_length / 6) or 1) if vertical else (round(min_line_length / 6) or 1, 1))
    mask_dotted = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, dotted_kernel)
    write_debug("mask_dotted", mask_dotted)

    # Apply masking on line length
    kernel_dims = (1, min_line_length or 1) if vertical else (min_line_length or 1, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
    final_mask = cv2.morphologyEx(mask_dotted, cv2.MORPH_OPEN, kernel, iterations=1)
    write_debug("mask_final", final_mask)

    # Get stats
    _, _, stats, _ = cv2.connectedComponentsWithStats(final_mask, 8, cv2.CV_32S)

    lines = []
    # Get relevant CC that correspond to lines
    for idx, stat in enumerate(stats):
        if idx == 0:
            continue

        # Get stats
        x, y, w, h, area = stat

        # Filter on aspect ratio
        if max(w, h) / min(w, h) < 5 and min(w, h) >= char_length:
            continue
        # Filter on length
        if max(w, h) < min_line_length:
            continue

        cropped = thresh[y:y+h, x:x+w]
        if w >= h:
            non_blank_pixels = np.where(np.sum(cropped, axis=0) > 0)
            line_rows = np.where((np.sum(cropped, axis=1) / 255) >= 0.5 * w)

            if len(line_rows[0]) == 0:
                continue

            line = Line(x1=x + np.min(non_blank_pixels),
                        y1=y + round(np.mean(line_rows)),
                        x2=x + np.max(non_blank_pixels),
                        y2=y + round(np.mean(line_rows)),
                        thickness=np.max(line_rows) - np.min(line_rows) + 1)
        else:
            non_blank_pixels = np.where(np.sum(cropped, axis=1) > 0)
            line_cols = np.where((np.sum(cropped, axis=0) / 255) >= 0.5 * h)

            if len(line_cols[0]) == 0:
                continue

            line = Line(x1=x + round(np.mean(line_cols)),
                        y1=y + np.min(non_blank_pixels),
                        x2=x + round(np.mean(line_cols)),
                        y2=y + np.max(non_blank_pixels),
                        thickness=np.max(line_cols) - np.min(line_cols) + 1)
        lines.append(line)

    return lines


def detect_lines(
    img: np.ndarray,
    contours: Optional[list[Cell]],
    char_length: Optional[float],
    min_line_length: Optional[float],
    debug_dir: Optional[str | Path] = None,
) -> (list[Line], list[Line]):
    """
    Detect horizontal and vertical rows on image
    :param img: image array
    :param contours: list of image contours as cell objects
    :param char_length: average character length
    :param min_line_length: minimum line length
    :param debug_dir: optional directory to dump debug images
    :return: horizontal and vertical rows
    """
    debug_dir_path = Path(debug_dir) if debug_dir is not None else None
    if debug_dir_path is not None:
        debug_dir_path.mkdir(parents=True, exist_ok=True)

    def write_debug(name: str, image: np.ndarray, rgb: bool = False) -> None:
        if debug_dir_path is None:
            return
        if image.dtype == np.bool_:
            image = image.astype(np.uint8) * 255
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 3 and rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(debug_dir_path / f"{name}.png"), image)

    write_debug("start_rgb", img, rgb=True)

    # Grayscale and blurring
    blur = cv2.bilateralFilter(img, 3, 40, 80)
    gray = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    write_debug("blur_rgb", blur, rgb=True)
    write_debug("gray", gray)

    # Apply laplacian and filter image
    laplacian = cv2.Laplacian(src=gray, ksize=3, ddepth=cv2.CV_64F)
    edge_img = cv2.convertScaleAbs(laplacian)
    write_debug("laplacian_abs", edge_img)

    # Remove contours and convert to binary image
    edge_wo_contours = edge_img.copy()
    if contours:
        contours_overlay = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
        for c in contours:
            cv2.rectangle(contours_overlay, (c.x1, c.y1), (c.x2 - 1, c.y2 - 1), (0, 0, 255), 1)
            edge_wo_contours[c.y1 - 1:c.y2 + 1, c.x1 - 1:c.x2 + 1] = 0
        write_debug("edge_contours_overlay", contours_overlay)
    write_debug("edge_no_contours", edge_wo_contours)
    binary_img = 255 * (edge_wo_contours >= min(2.5 * np.mean(edge_wo_contours), np.max(edge_wo_contours))).astype(np.uint8)
    write_debug("binary_edges", binary_img)

    # Detect lines
    h_lines = identify_straight_lines(thresh=binary_img,
                                      min_line_length=min_line_length,
                                      char_length=char_length,
                                      vertical=False,
                                      debug_dir=debug_dir_path,
                                      debug_prefix="h")
    v_lines = identify_straight_lines(thresh=binary_img,
                                      min_line_length=min_line_length,
                                      char_length=char_length,
                                      vertical=True,
                                      debug_dir=debug_dir_path,
                                      debug_prefix="v")

    if debug_dir_path is not None:
        overlay = img.copy()
        for line in h_lines:
            cv2.line(overlay, (line.x1, line.y1), (line.x2, line.y2), (0, 255, 0), 1)
        for line in v_lines:
            cv2.line(overlay, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 0), 1)
        write_debug("lines_overlay", overlay, rgb=True)

    return h_lines, v_lines
