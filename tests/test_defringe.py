import numpy as np

from app.whitening import defringe_near_white


def test_tinted_near_white_becomes_white():
    # Light pink-ish near-white (spill) in BGR
    img = np.full((20, 20, 3), (235, 240, 255), np.uint8)
    out = defringe_near_white(img)
    assert np.all(out >= 250)


def test_dark_subject_preserved():
    img = np.full((20, 20, 3), (30, 40, 50), np.uint8)
    out = defringe_near_white(img)
    np.testing.assert_array_equal(out, img)
