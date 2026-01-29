
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from img2table.tables.objects.cell import Cell


def get_image_elements(
    thresh: np.ndarray,
    char_length: float,
    debug_dir: Optional[str | Path] = None,
) -> list[Cell]:
    """
    Identify image elements
    :param thresh: thresholded image array
    :param char_length: average character length
    :param debug_dir: optional directory to dump debug images
    :param median_line_sep: median line separation
    :return: list of image elements
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
        cv2.imwrite(str(debug_dir_path / f"{name}.png"), image)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if debug_dir_path is not None:
        contour_vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_vis, cnts, -1, (0, 255, 255), 1)
        write_debug("image_elements_contours", contour_vis)

    # Get list of contours
    elements = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if ((min(h, w) >= 0.5 * char_length and max(h, w) >= char_length)
                or (w / h >= 2 and 0.5 * char_length <= w <= 1.5 * char_length)):
            elements.append(Cell(x1=x, y1=y, x2=x + w, y2=y + h))

    return elements
