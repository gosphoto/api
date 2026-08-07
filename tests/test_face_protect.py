import numpy as np

from app.face_protect import (
    _FACE_OVAL_IDX,
    _LEFT_EYE,
    _RIGHT_EYE,
    _similarity_from_eyes,
    apply_face_protect,
    face_protect_mask,
)


def test_face_protect_mask_none_without_landmarks(monkeypatch):
    import app.face_protect as fp

    monkeypatch.setattr(fp, "_landmarks_xy", lambda bgr: None)
    assert face_protect_mask(np.zeros((40, 40, 3), np.uint8)) is None


def test_face_protect_mask_covers_center_not_corners(monkeypatch):
    import app.face_protect as fp

    pts = np.zeros((478, 2), np.float32)
    pts[:] = [50.0, 50.0]
    for i, idx in enumerate(_FACE_OVAL_IDX):
        ang = 2 * np.pi * i / len(_FACE_OVAL_IDX)
        pts[idx] = [50 + 20 * np.cos(ang), 50 + 26 * np.sin(ang)]
    monkeypatch.setattr(fp, "_landmarks_xy", lambda bgr: pts)
    m = face_protect_mask(np.zeros((100, 100, 3), np.uint8))
    assert m is not None
    assert float(m[50, 50]) > 0.5
    assert float(m[2, 2]) < 0.05
    assert float(m[90, 50]) < 0.2


def test_similarity_from_eyes_maps_midpoint():
    src = np.zeros((478, 2), np.float32)
    dst = np.zeros((478, 2), np.float32)
    src[_LEFT_EYE] = [10, 20]
    src[_RIGHT_EYE] = [30, 20]
    dst[_LEFT_EYE] = [100, 200]
    dst[_RIGHT_EYE] = [140, 200]
    M = _similarity_from_eyes(src, dst)
    assert M is not None
    mid = np.array([20.0, 20.0, 1.0], dtype=np.float32)
    got = M @ mid
    assert abs(got[0] - 120.0) < 1.0
    assert abs(got[1] - 200.0) < 1.0


def test_protect_pastes_original_face_after_align(monkeypatch):
    import app.face_protect as fp

    pts = np.zeros((478, 2), np.float32)
    pts[:] = [50.0, 50.0]
    for i, idx in enumerate(_FACE_OVAL_IDX):
        ang = 2 * np.pi * i / len(_FACE_OVAL_IDX)
        pts[idx] = [50 + 20 * np.cos(ang), 50 + 26 * np.sin(ang)]
    pts[_LEFT_EYE] = [40, 45]
    pts[_RIGHT_EYE] = [60, 45]

    def fake_lm(bgr):
        # Same landmarks for both → align is near-identity
        return pts.copy()

    monkeypatch.setattr(fp, "_landmarks_xy", fake_lm)
    original = np.zeros((100, 100, 3), np.uint8)
    original[:] = (10, 20, 30)
    edited = np.zeros((100, 100, 3), np.uint8)
    edited[:] = (200, 200, 200)
    out, ok, meta = apply_face_protect(original, edited)
    assert ok
    assert "eyes" in meta["align"]
    assert int(out[50, 50, 0]) < 40
    assert int(out[2, 2, 0]) > 150
