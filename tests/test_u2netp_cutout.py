"""Silhouette cutout smoke (skips if ONNX model missing)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from app import config
from app.bg import _composite_on_white, _silhouette_alpha, white_background_local


def test_silhouette_alpha_keeps_core():
    conf = np.zeros((64, 64), np.float32)
    conf[16:48, 16:48] = 1.0
    face = np.zeros((64, 64), np.uint8)
    face[20:40, 20:40] = 255
    out = _silhouette_alpha(conf, face, thr=0.5, close_k=9, erode=1, feather=0.8)
    assert out[32, 32] > 0.85
    assert out[2, 2] < 0.05


def test_composite_white_corners():
    bgr = np.full((40, 40, 3), 40, np.uint8)
    bgr[10:30, 10:30] = (20, 40, 180)
    mask = np.zeros((40, 40), np.float32)
    mask[10:30, 10:30] = 1.0
    out = _composite_on_white(bgr, mask)
    assert np.all(out[0, 0] == 255)
    assert out[20, 20, 2] > 100


@pytest.mark.skipif(
    not Path(config.SILUETA_MODEL_PATH).is_file()
    and not Path(config.U2NETP_MODEL_PATH).is_file(),
    reason="no ONNX cutout model",
)
def test_onnx_cutout_whitens_corners(monkeypatch):
    backend = "silueta" if Path(config.SILUETA_MODEL_PATH).is_file() else "u2netp"
    monkeypatch.setattr(config, "EDIT_CUTOUT", backend)
    monkeypatch.setattr(config, "MIN_PROCESS_SIDE", 200)
    bgr = np.full((256, 192, 3), (40, 180, 40), np.uint8)
    cv2.ellipse(bgr, (96, 110), (55, 80), 0, 0, 360, (60, 80, 160), -1)
    out = white_background_local(bgr)
    assert out.shape == bgr.shape
    assert int(out[5, 5].mean()) >= 250
