"""Soft camera_too_close warning from face_height_frac — never fails gate."""

from __future__ import annotations

from app.gate_warnings import soft_warnings_for_face_height


def test_close_face_emits_camera_too_close_warning():
    # Roman-class close selfie (~0.381) vs default 0.37
    warnings, message = soft_warnings_for_face_height(0.381, threshold=0.37)
    assert warnings == ["camera_too_close"]
    assert message and "Госуслуги" in message
    assert "искажен" in message
    assert "метра" in message


def test_distant_face_has_no_warning():
    warnings, message = soft_warnings_for_face_height(0.30, threshold=0.37)
    assert warnings == []
    assert message is None


def test_threshold_boundary_is_exclusive():
    """Equal to threshold → no warn; just above → warn."""
    at, _ = soft_warnings_for_face_height(0.37, threshold=0.37)
    above, _ = soft_warnings_for_face_height(0.3701, threshold=0.37)
    assert at == []
    assert above == ["camera_too_close"]
