import numpy as np

from app.face_restore import _face_mask


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
    # expand to look like denser landmarks
    pts = np.vstack([pts, pts + 1.0, pts - 1.0])
    mask = _face_mask((100, 100), pts)
    assert mask.shape == (100, 100)
    assert float(mask.max()) > 0.5
    assert float(mask[0, 0]) < 0.2
