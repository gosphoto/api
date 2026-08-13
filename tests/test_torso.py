"""Unit tests for torso / resume-offer decision (no MediaPipe model required)."""

from app.torso import decide_torso_ok


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
