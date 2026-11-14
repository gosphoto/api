"""Default face-size knob: -10% vs PASSPORT_FACE_RATIO (Gosuslugi large-face)."""

from app import config


def test_face_size_experiment_default_is_slightly_smaller():
    assert abs(float(config.FACE_SIZE_EXPERIMENT) - 0.90) < 1e-9
    assert abs(config.crop_face_ratio_aim() - 0.675) < 1e-9
    assert config.PASSPORT_FACE_RATIO == 0.75
    assert abs(float(config.FACE_RATIO_MIN) - 0.65) < 1e-9
    assert abs(float(config.HEAD_HEIGHT_MM_MIN) - 30.0) < 1e-9
