"""Detect leftover wall on the hair silhouette, then flood it to #FFFFFF.

Live miss: gosphoto.ru/result/45f4367d18a19c9e1fcc05a0a000289f
Detector must stay off on other portraits; fill runs only when it fires.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.whitening import (
    _blur_hair_wall_spill,
    _is_hair_wall_spill_case,
    force_white_background,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "whitening" / "cyan_hair_fringe.jpg"
GRAY = Path(__file__).resolve().parent / "fixtures" / "whitening" / "gray_studio_passport.jpg"
CROP = Path(__file__).resolve().parent / "fixtures" / "crop_regression"


def test_detects_her_cool_hair_wall_fringe():
    bgr = cv2.imread(str(FIX))
    assert bgr is not None
    assert _is_hair_wall_spill_case(bgr, chin_y=569) is True


def test_skips_bald_elder_white_in():
    bgr = cv2.imread(str(CROP / "bald_elder" / "white_in.jpg"))
    assert bgr is not None
    assert _is_hair_wall_spill_case(bgr, chin_y=430) is False


def test_skips_high_hair_girl_white_in():
    bgr = cv2.imread(str(CROP / "high_hair_girl" / "white_in.jpg"))
    assert bgr is not None
    assert _is_hair_wall_spill_case(bgr, chin_y=923) is False


def test_skips_high_hair_man_white_in():
    bgr = cv2.imread(str(CROP / "high_hair_man" / "white_in.jpg"))
    assert bgr is not None
    assert _is_hair_wall_spill_case(bgr, chin_y=962) is False


def test_skips_clean_synthetic_hair_on_white():
    h, w = 80, 80
    bgr = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    core = (xx - 40) ** 2 + (yy - 28) ** 2 <= 12**2
    bgr[core] = (40, 50, 70)
    assert _is_hair_wall_spill_case(bgr, chin_y=55) is False
    np.testing.assert_array_equal(_blur_hair_wall_spill(bgr, chin_y=55), bgr)


def test_detects_synthetic_cool_wall_ring():
    h, w = 200, 160
    bgr = np.full((h, w, 3), 255, np.uint8)
    yy, xx = np.ogrid[:h, :w]
    core = (xx - 80) ** 2 + (yy - 70) ** 2 <= 36**2
    bgr[core] = (40, 50, 70)
    ring = ((xx - 80) ** 2 + (yy - 70) ** 2 <= 38**2) & (~core)
    bgr[ring] = (248, 242, 236)  # BGR cool leftover wall
    assert _is_hair_wall_spill_case(bgr, chin_y=130) is True


def test_blur_is_noop_when_case_skips():
    bgr = cv2.imread(str(CROP / "bald_elder" / "white_in.jpg"))
    assert bgr is not None
    out = _blur_hair_wall_spill(bgr, chin_y=430)
    np.testing.assert_array_equal(out, bgr)


def test_blur_softens_her_fringe_without_eating_shirt_or_face():
    bgr = cv2.imread(str(FIX))
    assert bgr is not None
    out = force_white_background(bgr, tol=55)
    h, w = bgr.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hair = gray < 140
    hair[int(0.55 * h) :, :] = False
    white = gray >= 250
    band = (
        (cv2.dilate(white.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        & (cv2.dilate(hair.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        & (~hair)
        & (gray >= 210)
        & (gray < 252)
    )
    B, G, R = cv2.split(bgr)
    cool = band & (B.astype(int) + 2 >= R.astype(int))
    assert int(cool.sum()) > 50
    chroma_in = np.linalg.norm(
        bgr[cool].astype(np.float32) - bgr[cool].astype(np.float32).mean(1, keepdims=True),
        axis=1,
    ).mean()
    chroma_out = np.linalg.norm(
        out[cool].astype(np.float32) - out[cool].astype(np.float32).mean(1, keepdims=True),
        axis=1,
    ).mean()
    assert chroma_out < chroma_in - 0.3, (chroma_in, chroma_out)

    y1, y2 = int(0.78 * h), int(0.95 * h)
    x1, x2 = int(0.32 * w), int(0.68 * w)
    shirt = np.abs(
        bgr[y1:y2, x1:x2].astype(np.int16) - out[y1:y2, x1:x2].astype(np.int16)
    )
    assert float(shirt.mean()) < 1.0, shirt.mean()

    fy1, fy2 = int(0.38 * h), int(0.55 * h)
    fx1, fx2 = int(0.38 * w), int(0.62 * w)
    face = np.abs(
        bgr[fy1:fy2, fx1:fx2].astype(np.int16) - out[fy1:fy2, fx1:fx2].astype(np.int16)
    )
    assert float(face.mean()) < 1.0, face.mean()

    dark = gray < 80
    np.testing.assert_array_equal(out[dark], bgr[dark])


def test_gray_studio_still_bleaches_plate():
    bgr = cv2.imread(str(GRAY))
    assert bgr is not None
    out = force_white_background(bgr, tol=55)
    from app.whitening import corner_whiteness

    assert corner_whiteness(out)["white_ok"] is True
