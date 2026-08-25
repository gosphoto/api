"""Route Riverflow Pro only for messy hair on a light, sharp selfie."""

from __future__ import annotations

import cv2
import numpy as np

from app.edit_router import GEMINI_FLASH_IMAGE, RIVERFLOW_PRO, choose_edit_model


def _canvas(bg, size=(400, 300)) -> np.ndarray:
    h, w = size
    return np.full((h, w, 3), bg, dtype=np.uint8)


def _paint_head(bgr: np.ndarray, *, messy: bool) -> None:
    h, w = bgr.shape[:2]
    cx, cy = w // 2, int(0.38 * h)
    axes = (int(0.16 * w), int(0.22 * h))
    cv2.ellipse(bgr, (cx, cy), axes, 0, 0, 360, (40, 50, 70), -1)
    # torso so the frame still looks like a portrait
    cv2.rectangle(
        bgr,
        (int(0.28 * w), int(0.52 * h)),
        (int(0.72 * w), h - 8),
        (50, 60, 80),
        -1,
    )
    if not messy:
        return
    rng = np.random.default_rng(7)
    for _ in range(90):
        ang = float(rng.uniform(200, 340))
        rad = float(rng.uniform(0.85, 1.25))
        x = int(cx + rad * axes[0] * np.cos(np.deg2rad(ang)))
        y = int(cy + rad * axes[1] * np.sin(np.deg2rad(ang)))
        x2 = int(x + rng.integers(-18, 19))
        y2 = int(y + rng.integers(-22, 6))
        cv2.line(bgr, (x, y), (x2, y2), (25, 30, 40), 1)


def _sharpen_texture(bgr: np.ndarray) -> np.ndarray:
    """Keep Laplacian above the Pro quality floor without changing silhouette."""
    noise = np.random.default_rng(1).integers(0, 18, bgr.shape, dtype=np.uint8)
    return cv2.add(bgr, noise)


def test_messy_hair_light_sharp_uses_pro(monkeypatch):
    import app.edit_router as router

    monkeypatch.setattr(router.config, "EDIT_ROUTE_PRO_ON_MESSY_HAIR", True)
    bgr = _canvas((232, 228, 224))
    _paint_head(bgr, messy=True)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(bgr)
    assert d.use_pro is True, d.as_dict()
    assert d.model == RIVERFLOW_PRO
    assert d.reason in ("messy_hair_light_sharp", "light_bg_low_contrast")


def test_neat_hair_light_sharp_uses_gemini(monkeypatch):
    import app.edit_router as router

    monkeypatch.setattr(router.config, "EDIT_ROUTE_PRO_ON_MESSY_HAIR", True)
    bgr = _canvas((250, 250, 250))
    _paint_head(bgr, messy=False)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(bgr)
    assert d.use_pro is False, d.as_dict()
    assert d.model == GEMINI_FLASH_IMAGE
    assert d.reason == "hair_neat"


def test_light_bg_low_contrast_neat_uses_pro(monkeypatch):
    import app.edit_router as router

    monkeypatch.setattr(router.config, "EDIT_ROUTE_PRO_ON_MESSY_HAIR", False)
    bgr = _canvas((188, 188, 188))
    _paint_head(bgr, messy=False)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(
        bgr,
        readiness={"reasons": ["bg_not_white"], "scores": {"corner_white_ok": False}},
    )
    assert d.use_pro is True, d.as_dict()
    assert d.model == RIVERFLOW_PRO
    assert d.reason == "light_bg_low_contrast"


def test_messy_hair_dark_bg_uses_gemini(monkeypatch):
    import app.edit_router as router

    monkeypatch.setattr(router.config, "EDIT_ROUTE_PRO_ON_MESSY_HAIR", True)
    bgr = _canvas((36, 40, 48))
    _paint_head(bgr, messy=True)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(bgr)
    assert d.use_pro is False, d.as_dict()
    assert d.model == GEMINI_FLASH_IMAGE
    assert d.reason == "bg_not_light"


def test_messy_hair_light_blurry_uses_gemini(monkeypatch):
    import app.edit_router as router

    monkeypatch.setattr(router.config, "EDIT_ROUTE_PRO_ON_MESSY_HAIR", True)
    bgr = _canvas((232, 228, 224))
    _paint_head(bgr, messy=True)
    bgr = cv2.GaussianBlur(bgr, (21, 21), 0)
    d = choose_edit_model(bgr)
    assert d.use_pro is False, d.as_dict()
    assert d.model == GEMINI_FLASH_IMAGE
    assert d.reason == "input_not_sharp"


def test_flag_off_dark_bg_uses_gemini(monkeypatch):
    import app.edit_router as router

    monkeypatch.setattr(router.config, "EDIT_ROUTE_PRO_ON_MESSY_HAIR", False)
    bgr = _canvas((36, 40, 48))
    _paint_head(bgr, messy=True)
    bgr = _sharpen_texture(bgr)
    d = choose_edit_model(bgr)
    assert d.use_pro is False
    assert d.model == GEMINI_FLASH_IMAGE
    assert d.reason == "route_disabled"
