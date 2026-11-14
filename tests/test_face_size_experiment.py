"""Default face-size knob: full PASSPORT_FACE_RATIO (no -5% shrink)."""

from app import config


def test_face_size_experiment_default_is_full_ratio():
    assert abs(float(config.FACE_SIZE_EXPERIMENT) - 1.0) < 1e-9
    assert abs(config.crop_face_ratio_aim() - 0.75) < 1e-9
    assert config.PASSPORT_FACE_RATIO == 0.75
