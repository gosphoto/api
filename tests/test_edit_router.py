"""Route Riverflow Pro when the wall is light and the input is sharp."""

from __future__ import annotations

import cv2
import numpy as np

from app.edit_router import GEMINI_FLASH_IMAGE, RIVERFLOW_PRO, choose_edit_model


def _canvas(bg, size=(400, 300)) -> np.ndarray:
    h, w = size
    return np.full((h, w, 3), bg, dtype=np.uint8)


def _paint_head(bgr: np.ndarray) -> None:
    h, w = bgr.shape[:2]
    cx, cy = w // 2, int(0.38 * h)
    axes = (int(0.16 * w), int(0.22 * h))
    cv2.ellipse(bgr, (cx, cy), axes, 0, 0, 360, (40, 50, 70), -1)
    cv2.rectangle(
        bgr,
        (int(0.28 * w), int(0.52 * h)),
        (int(0.72 * w), h - 8),
        (50, 60, 80),
        -1,
    )


def _sharpen_texture(bgr: np.ndarray) -> np.ndarray:
    """Keep Laplacian above the Pro quality floor without changing silhouette."""
    noise = np.random.default_rng(1).integers(0, 18, bgr.shape, dtype=np.uint8)
    return cv2.add(bgr, noise)


def test_neat_hair_light_sharp_uses_pro():
    bgr = _canvas((232, 228, 224))
    _paint_head(bgr)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(bgr)
    assert d.use_pro is True, d.as_dict()
    assert d.model == RIVERFLOW_PRO
    assert d.reason == "light_sharp"
    assert "messy_ok" not in d.scores


def test_dark_bg_uses_gemini():
    bgr = _canvas((36, 40, 48))
    _paint_head(bgr)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(bgr)
    assert d.use_pro is False, d.as_dict()
    assert d.model == GEMINI_FLASH_IMAGE
    assert d.reason == "bg_not_light"


def test_light_blurry_uses_gemini():
    bgr = _canvas((232, 228, 224))
    _paint_head(bgr)
    bgr = cv2.GaussianBlur(bgr, (21, 21), 0)
    d = choose_edit_model(bgr)
    assert d.use_pro is False, d.as_dict()
    assert d.model == GEMINI_FLASH_IMAGE
    assert d.reason == "input_not_sharp"


def test_flag_off_always_gemini(monkeypatch):
    import app.edit_router as router

    monkeypatch.setattr(router.config, "EDIT_ROUTE_PRO_ON_MESSY_HAIR", False)
    bgr = _canvas((232, 228, 224))
    _paint_head(bgr)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(bgr)
    assert d.use_pro is False
    assert d.model == GEMINI_FLASH_IMAGE
    assert d.reason == "route_disabled"
