"""Unit tests for upright orientation scoring (no image fixtures)."""

from __future__ import annotations

from app.gate import _pose_score


def test_pose_score_penalizes_not_upright():
    upright = _pose_score(0.0, 0.0, 0.0, True, face_center_y=0.5)
    inverted = _pose_score(0.0, 0.0, 0.0, False, face_center_y=0.5)
    assert inverted == upright + 180.0


def test_pose_score_prefers_higher_face_when_mesh_lies():
    """Regression: both orients report upright=True; lower face_cy must win."""
    inverted_looks_ok = _pose_score(2.9, 0.6, 3.4, True, face_center_y=0.673)
    truly_upright = _pose_score(0.7, 2.0, 9.5, True, face_center_y=0.447)
    assert truly_upright < inverted_looks_ok
