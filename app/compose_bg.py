"""Composite cutout images onto solid #FFFFFF."""

from __future__ import annotations

import numpy as np


def composite_on_white(bgr_or_bgra: np.ndarray) -> np.ndarray:
    """Return BGR uint8 on white. Opaque 3-channel images pass through."""
    if bgr_or_bgra.ndim != 3:
        raise ValueError("expected HWC image")
    _h, _w, c = bgr_or_bgra.shape
    if c == 3:
        return bgr_or_bgra.copy()
    if c != 4:
        raise ValueError(f"expected 3 or 4 channels, got {c}")

    # OpenCV IMREAD_UNCHANGED: BGRA
    bgr = bgr_or_bgra[:, :, :3].astype(np.float32)
    a = bgr_or_bgra[:, :, 3].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    out = bgr * a[:, :, None] + white * (1.0 - a[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)
