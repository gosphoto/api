import numpy as np

from app.whitening import _bleach_corner_chips, corner_whiteness, defringe_near_white


def test_tinted_near_white_becomes_white():
    img = np.full((20, 20, 3), (235, 240, 255), np.uint8)
    out = defringe_near_white(img)
    assert np.all(out >= 250)


def test_dark_subject_preserved():
    img = np.full((20, 20, 3), (30, 40, 50), np.uint8)
    out = defringe_near_white(img)
    np.testing.assert_array_equal(out, img)


def test_corner_bleach_skips_dark_clothing():
    """Blue shirt in bottom corners must not become white squares."""
    h, w = 100, 80
    img = np.full((h, w, 3), 255, np.uint8)
    # dark blue clothing in bottom corners
    img[-20:, :20] = (180, 80, 40)  # BGR blue-ish
    img[-20:, -20:] = (180, 80, 40)
    subject = np.zeros((h, w), np.uint8)
    subject[-25:, :25] = 255
    subject[-25:, -25:] = 255
    out = _bleach_corner_chips(img, subject, n=12)
    np.testing.assert_array_equal(out[-12:, :12], img[-12:, :12])
    np.testing.assert_array_equal(out[-12:, -12:], img[-12:, -12:])
    # top corners stay / become white (empty bg)
    assert np.all(out[:12, :12] == 255)


def test_corner_whiteness_ignores_clothing_bottom():
    h, w = 60, 60
    img = np.full((h, w, 3), 255, np.uint8)
    img[-15:, :15] = (200, 60, 30)
    img[-15:, -15:] = (200, 60, 30)
    info = corner_whiteness(img)
    assert info["white_ok"] is True
