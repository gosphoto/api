from __future__ import annotations

from app.crop_geom import crop_width_correction, nudge_crop_off_bottom


def test_nudge_crop_off_bottom_no_overflow():
    top, shift = nudge_crop_off_bottom(100.0, 800.0, 1080.0)
    assert top == 100.0
    assert shift == 0.0


def test_nudge_crop_off_bottom_shifts_up():
    top, shift = nudge_crop_off_bottom(239.1, 876.2, 1080.0)
    assert abs(shift - (239.1 + 876.2 - 1080.0)) < 1e-9
    assert abs((top + 876.2) - 1080.0) < 1e-9
    assert top < 239.1


def test_auto_corr_gemini_3_4_vs_35x45():
    # 864×1184 is what Gemini Flash actually returns (kept_model_aspect).
    corr = crop_width_correction(864, 1184, 827, 1063, override="auto")
    expected = (827 / 1063) / (864 / 1184)
    assert abs(corr - expected) < 1e-6
    assert 1.06 < corr < 1.07


def test_auto_corr_already_passport_is_one():
    corr = crop_width_correction(827, 1063, 827, 1063, override="auto")
    assert abs(corr - 1.0) < 1e-9


def test_auto_corr_exact_3_4():
    corr = crop_width_correction(3, 4, 35, 45, override="auto")
    assert abs(corr - (35 / 45) / (3 / 4)) < 1e-9
    assert abs(corr - 1.037037) < 1e-5


def test_default_off_is_one_even_on_3_4():
    corr = crop_width_correction(864, 1184, 827, 1063)
    assert corr == 1.0
    corr = crop_width_correction(864, 1184, 827, 1063, override="1")
    assert corr == 1.0
    corr = crop_width_correction(864, 1184, 827, 1063, override="1.0")
    assert corr == 1.0


def test_explicit_override_clipped_to_max():
    corr = crop_width_correction(100, 1000, 827, 1063, override="auto", corr_max=1.12)
    assert corr == 1.12


def test_narrower_than_passport_never_below_one():
    # Source wider than 35×45 → do not stretch the face.
    corr = crop_width_correction(1000, 1000, 827, 1063, override="auto")
    assert corr == 1.0
