"""Step 1 — edit selfie: normalize + white studio background.

No passport crop here. Output is a cleaned full-frame portrait ready for /api/crop.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from . import config
from .bg import prepare_for_cutout, white_background_local
from .gate import _decode_image, _resize_max_side
from .openrouter import OpenRouterError, edit_selfie
from .whitening import force_white_background

log = logging.getLogger("gosphoto-gate")


def _gentle_face_light(bgr: np.ndarray) -> np.ndarray:
    """Mild local contrast on L channel — keeps identity, helps dull phone selfies."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    # blend so we don't overcook skin
    l_out = cv2.addWeighted(l, 0.55, l2, 0.45, 0)
    return cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)


def edit_selfie_local(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Local edit path: upscale → white bg → whitening → gentle light."""
    src = prepare_for_cutout(bgr)
    src = _resize_max_side(src, config.MAX_IMAGE_SIDE)
    edited = white_background_local(src)
    edited = force_white_background(edited, tol=52)
    edited = _gentle_face_light(edited)
    # whitening again after light tweak (CLAHE can tint near-white)
    edited = force_white_background(edited, tol=48)
    return edited, {
        "cutout": config.EDIT_CUTOUT or "mediapipe",
        "width": int(edited.shape[1]),
        "height": int(edited.shape[0]),
    }


def run_edit_stage(
    data: bytes,
    mime: str = "image/jpeg",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Stage 1 entry: bytes → edited BGR (white bg, not cropped).

    Strategy:
      1) local MediaPipe/rembg cutout (default, free, identity-safe)
      2) optional OpenRouter if EDIT_BACKEND=openrouter|auto and local fails/forced
    """
    backend = config.EDIT_BACKEND
    use_local = backend in ("local", "auto", "")
    use_or = backend in ("openrouter", "auto") and bool(config.OPENROUTER_API_KEY)
    local_err: Exception | None = None
    meta: dict[str, Any] = {"stage": "edit"}

    if use_local:
        try:
            src = _decode_image(data)
            if src is None:
                raise RuntimeError("decode_error")
            edited, local_meta = edit_selfie_local(src)
            meta.update(local_meta)
            meta["model"] = local_meta.get("cutout", "mediapipe")
            return edited, meta
        except Exception as e:
            local_err = e
            log.warning("Local edit failed: %s", e)
            if backend == "local":
                raise

    if use_or:
        try:
            raw = edit_selfie(data, mime=mime)
        except OpenRouterError:
            raise
        edited = _decode_image(raw)
        if edited is None:
            raise RuntimeError("Edited image decode failed")
        edited = force_white_background(edited, tol=52)
        meta.update(
            {
                "model": config.OPENROUTER_IMAGE_MODEL,
                "cutout": "openrouter",
                "width": int(edited.shape[1]),
                "height": int(edited.shape[0]),
            }
        )
        return edited, meta

    if local_err:
        raise local_err
    raise RuntimeError(f"No edit backend available (EDIT_BACKEND={backend})")
