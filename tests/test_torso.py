"""Unit tests for torso / resume-offer decision (no MediaPipe model required)."""

import math

from app.torso import decide_torso_ok, shoulder_roll_from_landmarks


def test_shoulder_roll_levels_tilted_line():
    from app.torso import normalize_roll_deg

    # Anatomical L on image-right; right shoulder lower in image → small roll
    r = shoulder_roll_from_landmarks(
        left_shoulder_x=0.7,
        left_shoulder_y=0.4,
        left_vis=0.9,
        right_shoulder_x=0.3,
        right_shoulder_y=0.5,
        right_vis=0.9,
        min_visibility=0.45,
        min_shoulder_width=0.12,
    )
    assert r is not None
    raw = math.degrees(math.atan2(0.1, -0.4))
    expected = normalize_roll_deg(raw)
    assert abs(r.deg - expected) < 1e-6
    assert abs(r.deg) < 90


def test_shoulder_roll_facing_camera_level_near_zero():
    # Anatomical left on image-right (typical selfie) — must NOT return ±180
    r = shoulder_roll_from_landmarks(
        left_shoulder_x=0.75,
        left_shoulder_y=0.45,
        left_vis=1.0,
        right_shoulder_x=0.25,
        right_shoulder_y=0.45,
        right_vis=1.0,
        min_visibility=0.45,
        min_shoulder_width=0.12,
    )
    assert r is not None
    assert abs(r.deg) < 1e-6


def test_normalize_roll_wraps_near_180():
    from app.torso import normalize_roll_deg

    assert abs(normalize_roll_deg(-178.9) - 1.1) < 1e-9
    assert abs(normalize_roll_deg(177.0) - (-3.0)) < 1e-9
    assert abs(normalize_roll_deg(45.0) - 45.0) < 1e-9


def test_shoulder_roll_none_when_level_narrow():
    assert (
        shoulder_roll_from_landmarks(
            left_shoulder_x=0.48,
            left_shoulder_y=0.4,
            left_vis=0.9,
            right_shoulder_x=0.52,
            right_shoulder_y=0.4,
            right_vis=0.9,
            min_visibility=0.45,
            min_shoulder_width=0.12,
        )
        is None
    )


def test_shoulder_roll_none_low_visibility():
    assert (
        shoulder_roll_from_landmarks(
            left_shoulder_x=0.3,
            left_shoulder_y=0.4,
            left_vis=0.1,
            right_shoulder_x=0.7,
            right_shoulder_y=0.45,
            right_vis=0.9,
            min_visibility=0.45,
            min_shoulder_width=0.12,
        )
        is None
    )


def test_shoulder_roll_horizontal_is_near_zero():
    r = shoulder_roll_from_landmarks(
        left_shoulder_x=0.3,
        left_shoulder_y=0.45,
        left_vis=0.9,
        right_shoulder_x=0.7,
        right_shoulder_y=0.45,
        right_vis=0.9,
        min_visibility=0.45,
        min_shoulder_width=0.12,
    )
    assert r is not None
    assert abs(r.deg) < 1e-9


def test_torso_ok_when_shoulders_visible_below_face():
    r = decide_torso_ok(
        nose_y=0.25,
        left_shoulder_x=0.35,
        left_shoulder_y=0.45,
        left_vis=0.9,
        right_shoulder_x=0.65,
        right_shoulder_y=0.46,
        right_vis=0.88,
        min_visibility=0.45,
        min_shoulder_drop=0.06,
        min_shoulder_width=0.12,
    )
    assert r.ok is True
    assert r.reason == "torso_ok"


def test_torso_rejects_low_visibility():
    r = decide_torso_ok(
        nose_y=0.25,
        left_shoulder_x=0.35,
        left_shoulder_y=0.45,
        left_vis=0.2,
        right_shoulder_x=0.65,
        right_shoulder_y=0.46,
        right_vis=0.9,
        min_visibility=0.45,
        min_shoulder_drop=0.06,
        min_shoulder_width=0.12,
    )
    assert r.ok is False
    assert r.reason == "shoulders_low_visibility"


def test_torso_rejects_face_crop():
    r = decide_torso_ok(
        nose_y=0.4,
        left_shoulder_x=0.4,
        left_shoulder_y=0.42,
        left_vis=0.9,
        right_shoulder_x=0.6,
        right_shoulder_y=0.43,
        right_vis=0.9,
        min_visibility=0.45,
        min_shoulder_drop=0.06,
        min_shoulder_width=0.12,
    )
    assert r.ok is False
    assert r.reason == "shoulders_too_close_to_face"


def test_torso_rejects_narrow_shoulders():
    r = decide_torso_ok(
        nose_y=0.2,
        left_shoulder_x=0.48,
        left_shoulder_y=0.4,
        left_vis=0.9,
        right_shoulder_x=0.52,
        right_shoulder_y=0.4,
        right_vis=0.9,
        min_visibility=0.45,
        min_shoulder_drop=0.06,
        min_shoulder_width=0.12,
    )
    assert r.ok is False
    assert r.reason == "shoulders_too_narrow"
