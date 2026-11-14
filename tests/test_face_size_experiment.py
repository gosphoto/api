"""Default face-size knob: -5% vs PASSPORT_FACE_RATIO (Gosuslugi large-face)."""

from app import config


def test_face_size_experiment_default_is_slightly_smaller():
    assert abs(float(config.FACE_SIZE_EXPERIMENT) - 0.95) < 1e-9
    assert abs(config.crop_face_ratio_aim() - 0.7125) < 1e-9
    assert config.PASSPORT_FACE_RATIO == 0.75
