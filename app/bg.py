"""Local person cutout → solid white background (no paid API)."""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2
import numpy as np
from rembg import new_session, remove

from . import config

log = logging.getLogger("gosphoto-gate")


@lru_cache(maxsize=1)
def _session():
    model = config.REMBG_MODEL
    log.info("Loading rembg session model=%s", model)
    return new_session(model)


def warmup_rembg() -> None:
    _session()


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


def white_background_local(bgr: np.ndarray) -> np.ndarray:
    """Replace background with #FFFFFF using rembg alpha matte."""
    bgr = prepare_for_cutout(bgr)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("encode_for_rembg_failed")

    # Keep alpha so we can clean fringing ourselves
    cut = remove(
        buf.tobytes(),
        session=_session(),
        bgcolor=None,
    )
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

    # Trim semi-transparent fringe (common halo on hair), then soften
    _, hard = cv2.threshold(alpha, 200, 255, cv2.THRESH_BINARY)
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    soft = cv2.GaussianBlur(hard, (5, 5), 0)

    a = soft.astype(np.float32) / 255.0
    white = np.full_like(rgb, 255, dtype=np.float32)
    composed = rgb.astype(np.float32) * a[:, :, None] + white * (1.0 - a[:, :, None])
    return np.clip(composed, 0, 255).astype(np.uint8)
