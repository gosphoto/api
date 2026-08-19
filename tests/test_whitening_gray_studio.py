"""Regression: Gemini gray studio must bleach to #FFFFFF.

Live miss: gosphoto.ru/result/45f4367d18a19c9e1fcc05a0a000289f
Gosuslugi: «Фон неоднотонный» — plate was ~BGR 241, not #FFFFFF.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.whitening import corner_whiteness, force_white_background

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
