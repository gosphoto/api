"""Color-decontaminate soft alpha edges before white composite (hair halo)."""

from __future__ import annotations

import numpy as np

from app.compose_bg import composite_on_white, decontaminate_alpha_colors


def _hair_disk_contaminated(*, radius: int = 20) -> np.ndarray:
    """Brown opaque core; soft rim RGB already blended with white (classic halo)."""
    h = w = 64
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - 32.0) ** 2 + (yy - 32.0) ** 2)
    hair = np.array([40.0, 55.0, 90.0], dtype=np.float32)  # BGR brown
    white = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    bgra = np.zeros((h, w, 4), dtype=np.float32)
    core = dist <= float(radius - 3)
    rim = (dist > float(radius - 3)) & (dist <= float(radius))
    # Soft alpha on rim; RGB wrongly stores hair* a + white*(1-a) already.
    a_rim = np.clip(1.0 - (dist - (radius - 3)) / 3.0, 0.05, 0.95)
    bgra[core, :3] = hair
    bgra[core, 3] = 255
    for c in range(3):
        bgra[rim, c] = hair[c] * a_rim[rim] + white[c] * (1.0 - a_rim[rim])
    bgra[rim, 3] = a_rim[rim] * 255.0
    return np.clip(bgra, 0, 255).astype(np.uint8)


def test_decontaminate_restores_rim_toward_hair_color():
    bgra = _hair_disk_contaminated()
    fixed = decontaminate_alpha_colors(bgra)
    yy, xx = np.ogrid[:64, :64]
    dist = np.sqrt((xx - 32.0) ** 2 + (yy - 32.0) ** 2)
    rim = (dist > 17) & (dist <= 20) & (bgra[:, :, 3] > 20) & (bgra[:, :, 3] < 250)
    assert rim.any()
    before = bgra[rim, :3].astype(np.float32).mean(axis=0)
    after = fixed[rim, :3].astype(np.float32).mean(axis=0)
    # Closer to brown (40,55,90) than the washed-white rim.
    before_err = np.linalg.norm(before - np.array([40.0, 55.0, 90.0]))
    after_err = np.linalg.norm(after - np.array([40.0, 55.0, 90.0]))
    assert after_err < before_err - 15.0
    # Alpha unchanged.
    np.testing.assert_array_equal(fixed[:, :, 3], bgra[:, :, 3])


def test_composite_after_decontaminate_less_white_than_raw():
    bgra = _hair_disk_contaminated()
    raw = composite_on_white(bgra)
    fixed = composite_on_white(decontaminate_alpha_colors(bgra))
    yy, xx = np.ogrid[:64, :64]
    dist = np.sqrt((xx - 32.0) ** 2 + (yy - 32.0) ** 2)
    rim = (dist > 17) & (dist <= 20)
    raw_l = raw[rim].astype(np.float32).mean()
    fixed_l = fixed[rim].astype(np.float32).mean()
    assert fixed_l < raw_l - 10.0


def test_opaque_bgr_passthrough():
    img = np.full((8, 8, 3), (10, 20, 30), np.uint8)
    out = decontaminate_alpha_colors(img)
    np.testing.assert_array_equal(out, img)


def test_fully_opaque_bgra_unchanged():
    bgra = np.zeros((6, 6, 4), np.uint8)
    bgra[:, :, :3] = (33, 44, 55)
    bgra[:, :, 3] = 255
    out = decontaminate_alpha_colors(bgra)
    np.testing.assert_array_equal(out, bgra)
