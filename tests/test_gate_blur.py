"""Blur gate must not fail a studio cutout because the white backdrop is flat."""

from __future__ import annotations

import cv2
import numpy as np

from app import config
from app.gate import _blur_score


def _studio_cutout(*, sharp: bool) -> np.ndarray:
    h, w = 1600, 1244
    bgr = np.full((h, w, 3), 255, dtype=np.uint8)
    y0, y1, x0, x1 = 80, 1100, 280, 964
    face = np.full((y1 - y0, x1 - x0, 3), (160, 175, 200), dtype=np.uint8)
    noise = np.random.default_rng(0).integers(0, 40, face.shape, dtype=np.uint8)
    face = cv2.add(face, noise)
    bgr[y0:y1, x0:x1] = face
    if not sharp:
        bgr = cv2.GaussianBlur(bgr, (21, 21), 0)
    return bgr


def _full_frame_laplacian(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def test_studio_white_backdrop_is_ignored():
    bgr = _studio_cutout(sharp=True)
    subject = _blur_score(bgr)
    whole = _full_frame_laplacian(bgr)
    assert subject > whole
    assert subject >= config.MIN_BLUR_VARIANCE


def test_really_blurry_studio_cutout_still_fails():
    bgr = _studio_cutout(sharp=False)
    assert _blur_score(bgr) < config.MIN_BLUR_VARIANCE
