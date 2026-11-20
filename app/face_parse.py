"""Local BiSeNet (yakhyo) face parsing — CelebAMask-HQ 19 classes."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config

log = logging.getLogger("gosphoto-gate")

# CelebAMask-HQ / yakhyo BiSeNet class ids
CLASS_BG = 0
CLASS_SKIN = 1
CLASS_L_BROW = 2
CLASS_R_BROW = 3
CLASS_L_EYE = 4
CLASS_R_EYE = 5
CLASS_NOSE = 10
CLASS_MOUTH = 11
CLASS_U_LIP = 12
CLASS_L_LIP = 13
CLASS_HAIR = 17

OVAL_CLASS_IDS = (
    CLASS_L_BROW,
    CLASS_R_BROW,
    CLASS_L_EYE,
    CLASS_R_EYE,
    CLASS_NOSE,
    CLASS_MOUTH,
    CLASS_U_LIP,
    CLASS_L_LIP,
)

# Distinct BGR colors for overlay (index = class id 0..18)
_PALETTE_BGR = np.array(
    [
        [0, 0, 0],
        [0, 200, 255],
        [0, 80, 255],
        [0, 120, 255],
        [255, 80, 0],
        [255, 120, 0],
        [180, 180, 0],
        [200, 100, 200],
        [200, 140, 200],
        [220, 180, 220],
        [0, 255, 0],
        [255, 0, 255],
        [255, 100, 255],
        [255, 150, 255],
        [160, 160, 160],
        [140, 140, 180],
        [100, 100, 100],
        [0, 0, 255],  # hair — red in BGR
        [40, 40, 200],
    ],
    dtype=np.uint8,
)

_INPUT_SIZE = (512, 512)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    path = Path(config.BISENET_MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"BiSeNet ONNX missing: {path}")
    log.info("Loading BiSeNet from %s", path)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def warmup() -> None:
    """Load ONNX once at process start (optional)."""
    try:
        _session()
    except FileNotFoundError as e:
        log.warning("BiSeNet warmup skipped: %s", e)


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, _INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
    return x


def parse_mask(bgr: np.ndarray) -> np.ndarray:
    """Return HxW uint8 CelebAMask-HQ class map (same size as bgr)."""
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    h, w = bgr.shape[:2]
    sess = _session()
    inp = sess.get_inputs()[0]
    out_names = [o.name for o in sess.get_outputs()]
    outputs = sess.run(out_names, {inp.name: _preprocess(bgr)})
    pred = outputs[0].squeeze(0).argmax(0).astype(np.uint8)
    if pred.shape[0] != h or pred.shape[1] != w:
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def overlay_bgr(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Colorize mask over BGR image."""
    color = _PALETTE_BGR[np.clip(mask, 0, len(_PALETTE_BGR) - 1)]
    return cv2.addWeighted(bgr, 1.0 - alpha, color, alpha, 0)


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("mask png encode failed")
    return bytes(buf)


def overlay_to_png_bytes(bgr: np.ndarray, mask: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", overlay_bgr(bgr, mask))
    if not ok:
        raise RuntimeError("overlay png encode failed")
    return bytes(buf)


def parse_with_artifacts(bgr: np.ndarray) -> dict[str, Any]:
    """Parse once; return mask ndarray + png bytes for overlay/mask."""
    mask = parse_mask(bgr)
    return {
        "mask": mask,
        "mask_png": mask_to_png_bytes(mask),
        "overlay_png": overlay_to_png_bytes(bgr, mask),
    }
