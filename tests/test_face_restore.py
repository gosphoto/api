import numpy as np

from app.face_restore import _FACE_OVAL_IDX, _face_mask


def test_face_mask_soft_peak_in_center():
    pts = np.array(
        [
            [40.0, 40.0],
            [60.0, 40.0],
            [50.0, 70.0],
            [50.0, 50.0],
        ],
        dtype=np.float32,
    )
    pts = np.vstack([pts, pts + 1.0, pts - 1.0])
    face, hair = _face_mask((100, 100), pts)
    assert face.shape == (100, 100)
    assert float(face.max()) > 0.5
    assert float(face[0, 0]) < 0.2
    assert float(hair.max()) >= 0.0


def test_face_mask_oval_stays_inside_skin_box():
    """Full 468-point mesh: face core must not spill into far corners."""
    rng = np.random.default_rng(0)
    pts = rng.normal(loc=[50.0, 50.0], scale=8.0, size=(478, 2)).astype(np.float32)
    for i, idx in enumerate(_FACE_OVAL_IDX):
        ang = 2 * np.pi * i / len(_FACE_OVAL_IDX)
        pts[idx] = [50 + 22 * np.cos(ang), 50 + 28 * np.sin(ang)]
    face, hair = _face_mask((100, 100), pts)
    assert float(face[50, 50]) > 0.5
    assert float(face[2, 2]) < 0.05
    assert float(face[2, 97]) < 0.05
    assert float((face + hair)[97, 2]) < 0.2
