"""Local person cutout → solid white background (no paid API).

Default: MediaPipe selfie segmenter (light, VPS-safe).
Optional: rembg if EDIT_CUTOUT=rembg and package installed.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config

log = logging.getLogger("gosphoto-gate")


def prepare_for_cutout(bgr: np.ndarray) -> np.ndarray:
    """Upscale tiny selfies so cutout + crop edges stay sharp."""
    h, w = bgr.shape[:2]
    side = max(h, w)
    target = config.MIN_PROCESS_SIDE
    if side >= target:
        return bgr
    scale = target / side
    return cv2.resize(
        bgr,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


@lru_cache(maxsize=1)
def _segmenter() -> vision.ImageSegmenter:
    path = Path(config.SELFIE_SEGMENTER_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"Selfie segmenter model missing: {path}")
    log.info("Loading MediaPipe selfie segmenter from %s", path)
    options = vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def warmup_cutout() -> None:
    backend = config.EDIT_CUTOUT
    if backend == "rembg":
        _rembg_session()
    else:
        _segmenter()


def _white_bg_mediapipe(bgr: np.ndarray) -> np.ndarray:
    bgr = prepare_for_cutout(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _segmenter().segment(mp_image)
    if not result.confidence_masks:
        raise RuntimeError("selfie_segmenter_no_mask")
    mask = result.confidence_masks[0].numpy_view().astype(np.float32)
    # Soft person matte; suppress weak fringe
    mask = np.clip((mask - 0.30) / 0.40, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    white = np.full_like(bgr, 255, dtype=np.float32)
    out = bgr.astype(np.float32) * mask[:, :, None] + white * (1.0 - mask[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _rembg_session():
    from rembg import new_session

    model = config.REMBG_MODEL
    log.info("Loading rembg session model=%s", model)
    return new_session(model)


def _white_bg_rembg(bgr: np.ndarray) -> np.ndarray:
    from rembg import remove

    bgr = prepare_for_cutout(bgr)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("encode_for_rembg_failed")
    cut = remove(buf.tobytes(), session=_rembg_session(), bgcolor=None)
    arr = np.frombuffer(cut, dtype=np.uint8)
    out = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if out is None:
        raise RuntimeError("rembg_decode_failed")
    if out.ndim == 2:
        rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        alpha = np.full(out.shape, 255, np.uint8)
    elif out.shape[2] == 4:
        rgb = out[:, :, :3]
        alpha = out[:, :, 3]
    else:
        return out
    _, hard = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    soft = cv2.GaussianBlur(hard, (5, 5), 0)
    a = soft.astype(np.float32) / 255.0
    white = np.full_like(rgb, 255, dtype=np.float32)
    composed = rgb.astype(np.float32) * a[:, :, None] + white * (1.0 - a[:, :, None])
    return np.clip(composed, 0, 255).astype(np.uint8)


def white_background_local(bgr: np.ndarray) -> np.ndarray:
    """Replace background with #FFFFFF."""
    backend = config.EDIT_CUTOUT
    if backend == "rembg":
        try:
            return _white_bg_rembg(bgr)
        except Exception as e:
            log.warning("rembg failed, falling back to mediapipe: %s", e)
    return _white_bg_mediapipe(bgr)


# Back-compat aliases used by main.py
def warmup_rembg() -> None:
    warmup_cutout()
