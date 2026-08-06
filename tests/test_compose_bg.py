import numpy as np

from app.compose_bg import composite_on_white


def test_opaque_bgr_unchanged_shape():
    img = np.full((10, 10, 3), (40, 80, 120), np.uint8)
    out = composite_on_white(img)
    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_fully_transparent_becomes_white():
    bgra = np.zeros((8, 8, 4), np.uint8)
    bgra[:, :, 0] = 0
    bgra[:, :, 1] = 0
    bgra[:, :, 2] = 255
    bgra[:, :, 3] = 0
    out = composite_on_white(bgra)
    assert np.all(out == 255)


def test_half_alpha_blends_toward_white():
    bgra = np.zeros((4, 4, 4), np.uint8)
    bgra[:, :, 0] = 0
    bgra[:, :, 1] = 0
    bgra[:, :, 2] = 0
    bgra[:, :, 3] = 128
    out = composite_on_white(bgra)
    assert 120 <= int(out[0, 0, 0]) <= 140
