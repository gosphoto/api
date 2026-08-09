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


def test_as_dict_shape():
    bgr = _portrait(bg=255)
    d = assess_readiness(bgr).as_dict()
    assert set(d) >= {"ready", "reason", "reasons", "scores", "source"}
