"""EXPERIMENT 2026-09-02: all heads 5% smaller. Rollback = FACE_SIZE_EXPERIMENT 1.0."""

from app import config


def test_face_size_experiment_is_five_percent():
    assert config.FACE_SIZE_EXPERIMENT == 0.95
    assert abs(config.crop_face_ratio_aim() - 0.75 * 0.95) < 1e-9
    assert config.PASSPORT_FACE_RATIO == 0.75
