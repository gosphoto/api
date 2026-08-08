"""Nano Banana white-bg edit. Output is a full-frame portrait ready for crop."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from . import config
from .bg import prepare_for_cutout
from .compose_bg import composite_on_white
from .gate import _decode_image, _resize_max_side
from .openrouter import edit_selfie_nano_banana
from .whitening import force_white_background

log = logging.getLogger("gosphoto-gate")


def _decode_any(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def _uniform_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    """Scale preserving aspect — never squash to another frame's HxW."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    return cv2.resize(
        bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def run_edit_nano_banana(
    data: bytes,
    mime: str = "image/jpeg",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Nano Banana Pro: single pass white-bg. No face-protect, no pass2."""
    src = _decode_image(data)
    if src is None:
        raise RuntimeError("decode_error")
    src_p = prepare_for_cutout(src)
    src_p = _resize_max_side(src_p, config.MAX_IMAGE_SIDE)

    ok, buf = cv2.imencode(".jpg", src_p, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("encode_for_nano_banana_failed")
    orig_jpg = buf.tobytes()

    raw = edit_selfie_nano_banana(orig_jpg, mime="image/jpeg")
    decoded = _decode_any(raw)
    if decoded is None:
        raise RuntimeError("Nano Banana decode failed")
    out = composite_on_white(decoded)
    # Keep model aspect (3:4). Squashing into src_p HxW stretches faces
    # when the selfie is square/landscape (e.g. 513×480).
    out = _uniform_max_side(out, config.MAX_IMAGE_SIDE)
    out = force_white_background(out, tol=48)

    return out, {
        "model": config.NANO_BANANA_MODEL,
        "cutout": "nano_banana",
        "face_protected": False,
        "passes": 1,
        "prompt": "gosuslugi_nano",
        "src_size": [int(src_p.shape[1]), int(src_p.shape[0])],
        "kept_model_aspect": True,
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    }
