from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from img2table.tables.objects.line import Line
from img2table.tables.objects.table import Table
from img2table.tables.processing.borderless_tables.layout.column_segments import segment_image_columns
from img2table.tables.processing.borderless_tables.layout.image_elements import get_image_elements
from img2table.tables.processing.borderless_tables.layout.rlsa import identify_text_mask
from img2table.tables.processing.borderless_tables.layout.table_segments import get_table_segments
from img2table.tables.processing.borderless_tables.model import TableSegment, ImageSegment


def segment_image(
    thresh: np.ndarray,
    lines: list[Line],
    char_length: float,
    median_line_sep: float,
    existing_tables: Optional[list[Table]] = None,
    debug_dir: Optional[str | Path] = None,
) -> list[TableSegment]:
    """
    Segment image and its elements
    :param thresh: threshold image array
    :param lines: list of Line objects of the image
    :param char_length: average character length
    :param median_line_sep: median line separation
    :param existing_tables: list of detected bordered tables
    :param debug_dir: optional directory to dump debug images
    :return: list of ImageSegment objects with corresponding elements
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

    write_debug("segment_thresh", thresh)

    # Identify text mask
    rlsa_debug = (debug_dir_path / "rlsa") if debug_dir_path is not None else None
    text_thresh = identify_text_mask(thresh=thresh,
                                     lines=lines,
                                     char_length=char_length,
                                     existing_tables=existing_tables,
                                     debug_dir=rlsa_debug)
    write_debug("segment_text_mask", text_thresh)

    # Identify image elements
    img_elements = get_image_elements(thresh=text_thresh,
                                      char_length=char_length)

    if len(img_elements) == 0:
        return []

    if debug_dir_path is not None:
        elements_overlay = cv2.cvtColor(text_thresh, cv2.COLOR_GRAY2BGR)
        for el in img_elements:
            cv2.rectangle(elements_overlay, (el.x1, el.y1), (el.x2 - 1, el.y2 - 1), (0, 255, 0), 1)
        write_debug("segment_elements", elements_overlay)

    # Identify column segments
    y_min, y_max = min([el.y1 for el in img_elements]), max([el.y2 for el in img_elements])
    image_segment = ImageSegment(x1=0, y1=y_min, x2=thresh.shape[1], y2=y_max, elements=img_elements)

    col_segments = segment_image_columns(image_segment=image_segment,
                                         char_length=char_length,
                                         lines=lines)

    if debug_dir_path is not None:
        col_overlay = cv2.cvtColor(text_thresh, cv2.COLOR_GRAY2BGR)
        for seg in col_segments:
            cv2.rectangle(col_overlay, (seg.x1, seg.y1), (seg.x2 - 1, seg.y2 - 1), (255, 0, 0), 1)
        write_debug("segment_columns", col_overlay)

    # Within each column, identify segments that can correspond to tables
    table_segments = [table_segment for col_segment in col_segments
                      for table_segment in get_table_segments(segment=col_segment,
                                                              char_length=char_length,
                                                              median_line_sep=median_line_sep)
                      ]

    if debug_dir_path is not None:
        table_overlay = cv2.cvtColor(text_thresh, cv2.COLOR_GRAY2BGR)
        for seg in table_segments:
            cv2.rectangle(table_overlay, (seg.x1, seg.y1), (seg.x2 - 1, seg.y2 - 1), (0, 128, 255), 2)
        write_debug("segment_tables", table_overlay)

    return table_segments
