"""10×15 cm print sheet with 4× 35×45 photos."""

import numpy as np

from app.print_sheet import (
    COPIES,
    PRINT_DPI,
    PRINT_HEIGHT_MM,
    PRINT_WIDTH_MM,
    encode_print_jpeg,
    make_print_sheet_bgr,
)


def test_print_sheet_size_and_copies():
    # Fake passport crop at digital size
    passport = np.full((1063, 827, 3), 240, dtype=np.uint8)
    passport[200:900, 200:600] = (40, 70, 140)
    sheet, meta = make_print_sheet_bgr(passport)

    assert meta["copies"] == COPIES == 4
    assert meta["size_cm"] == [10, 15]
    assert meta["dpi"] == PRINT_DPI == 300
    assert meta["width"] == round(PRINT_WIDTH_MM / 25.4 * PRINT_DPI)
    assert meta["height"] == round(PRINT_HEIGHT_MM / 25.4 * PRINT_DPI)
    assert sheet.shape == (meta["height"], meta["width"], 3)
    # Not a blank sheet — photos placed
    assert int(sheet.min()) < 250


def test_print_jpeg_encodes():
    passport = np.full((531, 413, 3), 255, dtype=np.uint8)
    sheet, _ = make_print_sheet_bgr(passport)
    jpeg = encode_print_jpeg(sheet)
    assert jpeg[:2] == b"\xff\xd8"
    assert len(jpeg) > 1000
