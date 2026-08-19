"""Regression: Gemini gray studio must bleach to #FFFFFF.

Live miss: gosphoto.ru/result/45f4367d18a19c9e1fcc05a0a000289f
Gosuslugi: «Фон неоднотонный» — plate was ~BGR 241, not #FFFFFF.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.whitening import _studio_plate_mask, corner_whiteness, force_white_background

FIX = Path(__file__).resolve().parent / "fixtures" / "whitening" / "gray_studio_passport.jpg"


def test_fixture_is_the_gray_studio_miss():
    bgr = cv2.imread(str(FIX))
    assert bgr is not None
    info = corner_whiteness(bgr)
    assert info["white_ok"] is False, info
    h, w = bgr.shape[:2]
    top = bgr[: max(8, int(0.08 * h)), :]
    assert float((top.min(axis=2) >= 245).mean()) < 0.2


def test_gray_studio_passport_bleaches_to_white():
    bgr = cv2.imread(str(FIX))
    assert bgr is not None
    out = force_white_background(bgr, tol=55)
    info = corner_whiteness(out)
    assert info["white_ok"] is True, info

    h, w = out.shape[:2]
    top = out[: max(8, int(0.08 * h)), :]
    assert float((top.min(axis=2) >= 245).mean()) >= 0.85, (
        f"top plate still gray: mean={top.reshape(-1, 3).mean(0)}"
    )

    # Side margins were ~241 while the top went to 255 — Gosuslugi reads that
    # as a non-uniform plate. Light pixels in top+side bands must bleach.
    band = 8
    border = np.zeros((h, w), dtype=bool)
    border[:band, :] = True
    border[:, :band] = True
    border[:, w - band :] = True
    border[-band:, :] = False  # jacket in bottom corners
    need = border & (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) >= 200)
    assert int(need.sum()) > 50
    assert float((out.min(axis=2)[need] >= 245).mean()) >= 0.9, (
        f"side plate still gray: left={out[:, :band].reshape(-1, 3).mean(0)}"
    )

    # Dark jacket / hair must stay dark (do not flood-fill the person).
    dark = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 80
    assert int(dark.sum()) > 100
    assert float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[dark].mean()) < 100

    # Gray studio behind the shoulders (below-chin side strips) must bleach.
    # Use flat near-white pixels only — a wide band also covers the shirt island.
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(luma, (5, 5), 0)
    local_std = np.sqrt(np.maximum(cv2.blur((luma - blur) ** 2, (5, 5)), 0.0))
    side = int(0.18 * w)
    y0 = int(0.88 * h)
    behind = np.zeros((h, w), dtype=bool)
    behind[y0:, :side] = True
    behind[y0:, w - side :] = True
    need = behind & (luma >= 220) & (local_std <= 8)
    assert int(need.sum()) > 200
    assert float((out.min(axis=2)[need] >= 245).mean()) >= 0.9, (
        f"gray left behind shoulders: mean={out[need].mean(0)}"
    )

    # White collar / shirt in the center must not be painted #FFFFFF.
    y1, y2 = int(0.72 * h), int(0.92 * h)
    x1, x2 = int(0.38 * w), int(0.62 * w)
    patch_in = bgr[y1:y2, x1:x2]
    patch_out = out[y1:y2, x1:x2]
    shirt = (cv2.cvtColor(patch_in, cv2.COLOR_BGR2GRAY) > 140) & (
        cv2.cvtColor(patch_in, cv2.COLOR_BGR2GRAY) < 235
    )
    assert int(shirt.sum()) > 50
    assert float((patch_out.min(axis=2)[shirt] >= 250).mean()) < 0.25, (
        "collar was bleached"
    )


def test_studio_plate_bleaches_side_strips_not_bottom_shirt():
    """Gray connected to the top plate must bleach; a shirt island on the
    bottom edge must not — even if a fat subject mask covers the whole frame.
    """
    h, w = 180, 120
    bgr = np.full((h, w, 3), 241, np.uint8)
    # dark jacket barrier
    bgr[90:, 28:92] = (40, 45, 55)
    # white shirt island touching the bottom, not the sides
    bgr[120:, 45:75] = (228, 232, 236)
    subject = np.full((h, w), 255, np.uint8)
    plate = _studio_plate_mask(bgr, subject, chin_y=95)
    assert plate[20, 10] == 255  # top studio
    assert plate[150, 8] == 255  # gray behind shoulders
    assert plate[150, w - 8] == 255
    assert plate[160, 60] == 0  # shirt island
