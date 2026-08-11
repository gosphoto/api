import numpy as np

from app.readiness import assess_readiness


def _portrait(*, bg, person=(40, 55, 80), size=(400, 300)) -> np.ndarray:
    h, w = size
    bgr = np.full((h, w, 3), bg, dtype=np.uint8)
    # torso + head blob (not landmark-perfect; readiness uses mask/corners)
    bgr[120:380, 90:210] = person
    bgr[60:160, 105:195] = person
    return bgr


def test_studio_white_ready():
    bgr = _portrait(bg=255)
    r = assess_readiness(bgr)
    assert r.ready is True
    assert r.reason == "studio_ready"
    assert r.scores["corner_white_ok"] is True


def test_gray_wall_not_ready():
    bgr = _portrait(bg=(180, 180, 180))
    r = assess_readiness(bgr)
    assert r.ready is False
    assert "bg_not_white" in r.reasons or r.reason == "bg_not_white"


def test_colored_wall_not_ready():
    bgr = _portrait(bg=(210, 190, 140))
    r = assess_readiness(bgr)
    assert r.ready is False


def test_white_with_shadow_band_not_ready():
    bgr = _portrait(bg=255)
    # soft dark vignette on left border — classic home wall shadow
    bgr[:, :40] = (210, 210, 210)
    r = assess_readiness(bgr)
    assert r.ready is False


def test_tight_studio_white_shirt_in_corners_still_ready():
    """Light tee only in bottom corner chips; plate must stay studio-ready."""
    h, w = 400, 300
    bgr = np.full((h, w, 3), 255, dtype=np.uint8)
    bgr[60:180, 90:210] = (40, 55, 80)
    bgr[160:380, 80:220] = (40, 55, 80)
    # Corner-chip sized light shirt (not whole border — that is subject-mask territory).
    bgr[-15:, :15] = (220, 222, 225)
    bgr[-15:, -15:] = (218, 220, 224)
    r = assess_readiness(bgr)
    assert r.scores["corner_white_ok"] is True, r.as_dict()
    assert r.ready is True, r.as_dict()
    assert r.reason == "studio_ready"


def test_as_dict_shape():
    bgr = _portrait(bg=255)
    d = assess_readiness(bgr).as_dict()
    assert set(d) >= {"ready", "reason", "reasons", "scores", "source"}
