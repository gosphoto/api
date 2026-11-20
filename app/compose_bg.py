"""Composite cutout images onto solid #FFFFFF."""

from __future__ import annotations

import cv2
import numpy as np


def decontaminate_alpha_colors(
    bgr_or_bgra: np.ndarray,
    *,
    radius: int = 2,
    soft_lo: float = 0.05,
    soft_hi: float = 0.98,
) -> np.ndarray:
    """Replace soft-edge RGB with nearby opaque foreground color.

    Classic cutout halo: semi-transparent hair pixels already contain a blend
    with the old (often white) backdrop. Sampling opaque neighbors recovers
    true hair color before compositing onto #FFFFFF (Photoshop-style
    Decontaminate / Remove White Matte intent).
    """
    if bgr_or_bgra.ndim != 3:
        raise ValueError("expected HWC image")
    if bgr_or_bgra.shape[2] == 3:
        return bgr_or_bgra.copy()
    if bgr_or_bgra.shape[2] != 4:
        raise ValueError(f"expected 3 or 4 channels, got {bgr_or_bgra.shape[2]}")

    out = bgr_or_bgra.copy()
    bgr = out[:, :, :3].astype(np.float32)
    a = out[:, :, 3].astype(np.float32) / 255.0
    opaque = a >= soft_hi
    soft = (a > soft_lo) & (a < soft_hi)
    if not soft.any() or not opaque.any():
        return out

    # Mean of opaque BGR in a local window (ignore non-opaque via mask).
    opaque_u8 = opaque.astype(np.uint8)
    k = 2 * max(1, int(radius)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Count of opaque neighbors (including self when opaque).
    count = cv2.filter2D(opaque_u8.astype(np.float32), -1, kernel.astype(np.float32))
    sum_bgr = np.stack(
        [
            cv2.filter2D(bgr[:, :, c] * opaque_u8, -1, kernel.astype(np.float32))
            for c in range(3)
        ],
        axis=2,
    )
    # Avoid div0; where no opaque neighbor, fall back to unpremultiply-vs-white.
    safe = np.maximum(count, 1.0)[:, :, None]
    neighbor = sum_bgr / safe
    has_neighbor = count >= 1.0

    white = 255.0
    a_safe = np.maximum(a, 1e-3)[:, :, None]
    unpremul = (bgr - white * (1.0 - a_safe)) / a_safe
    unpremul = np.clip(unpremul, 0.0, 255.0)

    replaced = np.where(has_neighbor[:, :, None], neighbor, unpremul)
    soft3 = soft[:, :, None]
    bgr = np.where(soft3, replaced, bgr)
    out[:, :, :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    return out


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
