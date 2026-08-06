"""u2netp cutout smoke (skips if model/onnx missing)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app import config
from app.bg import _composite_on_white, _soft_matte_from_confidence, white_background_local


def test_soft_matte_kills_weak_fringe():
    mask = np.zeros((64, 64), np.float32)
    mask[16:48, 16:48] = 1.0
    mask[14:50, 14:50] = np.maximum(mask[14:50, 14:50], 0.3)
    out = _soft_matte_from_confidence(mask)
    assert out[32, 32] > 0.9
    assert out[2, 2] < 0.05


def test_composite_white_corners():
    bgr = np.full((40, 40, 3), 40, np.uint8)
    bgr[10:30, 10:30] = (20, 40, 180)
    mask = np.zeros((40, 40), np.float32)
    mask[10:30, 10:30] = 1.0
    out = _composite_on_white(bgr, mask)
    assert np.all(out[0, 0] == 255)
    assert out[20, 20, 2] > 100  # red channel of BGR subject


@pytest.mark.skipif(
    not Path(config.U2NETP_MODEL_PATH).is_file(),
    reason="u2netp.onnx not present",
)
def test_u2netp_cutout_whitens_corners(monkeypatch):
    monkeypatch.setattr(config, "EDIT_CUTOUT", "u2netp")
    monkeypatch.setattr(config, "MIN_PROCESS_SIDE", 200)
    # synthetic person blob on green bg
    bgr = np.full((256, 192, 3), (40, 180, 40), np.uint8)
    cv2.ellipse(bgr, (96, 110), (55, 80), 0, 0, 360, (60, 80, 160), -1)
    out = white_background_local(bgr)
    assert out.shape == bgr.shape
    assert int(out[5, 5].mean()) >= 250
