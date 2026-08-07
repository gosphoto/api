"""Print sheet: 10×15 cm with 4 copies of 35×45 mm passport photo."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from . import config

# Lab print sheet (standard RU 10×15)
PRINT_WIDTH_MM = 100.0
PRINT_HEIGHT_MM = 150.0
PRINT_DPI = 300
GAP_MM = 2.0
COPIES = 4
COLS = 2
ROWS = 2


def _mm_to_px(mm: float, dpi: int = PRINT_DPI) -> int:
    return max(1, int(round(mm / 25.4 * dpi)))


def make_print_sheet_bgr(passport_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Tile passport crop 2×2 onto a white 10×15 cm @300 dpi sheet."""
    sheet_w = _mm_to_px(PRINT_WIDTH_MM)
    sheet_h = _mm_to_px(PRINT_HEIGHT_MM)
    photo_w = _mm_to_px(config.PASSPORT_WIDTH_MM)
    photo_h = _mm_to_px(config.PASSPORT_HEIGHT_MM)
    gap = _mm_to_px(GAP_MM)

    photo = cv2.resize(passport_bgr, (photo_w, photo_h), interpolation=cv2.INTER_AREA)
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    grid_w = COLS * photo_w + (COLS - 1) * gap
    grid_h = ROWS * photo_h + (ROWS - 1) * gap
    origin_x = max(0, (sheet_w - grid_w) // 2)
    origin_y = max(0, (sheet_h - grid_h) // 2)

    for row in range(ROWS):
        for col in range(COLS):
            x0 = origin_x + col * (photo_w + gap)
            y0 = origin_y + row * (photo_h + gap)
            x1, y1 = x0 + photo_w, y0 + photo_h
            if x1 > sheet_w or y1 > sheet_h:
                continue
            sheet[y0:y1, x0:x1] = photo

    metrics = {
        "width": sheet_w,
        "height": sheet_h,
        "dpi": PRINT_DPI,
        "size_cm": [10, 15],
        "copies": COPIES,
        "photo_mm": [config.PASSPORT_WIDTH_MM, config.PASSPORT_HEIGHT_MM],
        "photo_px": [photo_w, photo_h],
        "gap_mm": GAP_MM,
        "layout": f"{COLS}x{ROWS}",
    }
    return sheet, metrics


def encode_print_jpeg(bgr: np.ndarray, quality: int = 92) -> bytes:
    """JPEG for photo-lab print (300 dpi, no 300 KB cap)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=quality,
        dpi=(PRINT_DPI, PRINT_DPI),
        optimize=True,
    )
    return buf.getvalue()
