"""Hair outline symmetry helpers for light-background cutout fixes."""

from __future__ import annotations

import cv2
import numpy as np

from app.bg import adaptive_tol
from app.whitening import _symmetrize_hair_outline, measure_outline_symmetry


def test_adaptive_tol_lower_on_light_wall():
    light = np.full((200, 200, 3), 190, np.uint8)
    dark = np.full((200, 200, 3), 40, np.uint8)
    assert adaptive_tol(light) < adaptive_tol(dark)


def test_symmetrize_preserves_protruding_hair_blob():
    h, w = 120, 100
    bgr = np.full((h, w, 3), 255, np.uint8)
    bgr[25:85, 42:58] = (30, 35, 40)
    bgr[25:55, 58:72] = (30, 35, 40)
    out = _symmetrize_hair_outline(bgr, cx=50.0, chin_y=85, gray_thr=200)
    assert int((out[30:50, 58:72] < 250).sum()) >= int((bgr[30:50, 58:72] < 250).sum())


def test_measure_outline_symmetry_zero_on_blank():
    bgr = np.full((80, 80, 3), 255, np.uint8)
    assert measure_outline_symmetry(bgr) == 0.0
