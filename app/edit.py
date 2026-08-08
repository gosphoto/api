"""Step 1 — edit selfie: normalize + white studio background.

No passport crop here. Output is a cleaned full-frame portrait ready for /api/crop.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from . import config
from .bg import last_cutout_backend, prepare_for_cutout, white_background_local
from .compose_bg import composite_on_white
from .face_protect import apply_face_protect, face_protect_mask
from .gate import _decode_image, _resize_max_side
from .openrouter import edit_selfie, edit_selfie_nano_banana
from .whitening import force_white_background

log = logging.getLogger("gosphoto-gate")


def _gentle_light_outside_face(bgr: np.ndarray) -> np.ndarray:
    """Mild CLAHE only outside the face zone — never paste/align another frame."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    l_out = cv2.addWeighted(l, 0.55, l2, 0.45, 0)
    lit = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2BGR)
    protect = face_protect_mask(bgr)
    if protect is None:
        return bgr
    m = protect[:, :, None]
    return (bgr.astype(np.float32) * m + lit.astype(np.float32) * (1.0 - m)).astype(
        np.uint8
    )


def edit_selfie_local(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Local cutout on white. No face paste/align — that shifts features off the head."""
    src = prepare_for_cutout(bgr)
    src = _resize_max_side(src, config.MAX_IMAGE_SIDE)
    edited = white_background_local(src)
    edited = force_white_background(edited, tol=48)
    edited = _gentle_light_outside_face(edited)
    edited = force_white_background(edited, tol=45)
    return edited, {
        "cutout": last_cutout_backend,
        "face_protected": True,
        "face_protect": {
            "model": "local_person",
            "applied": True,
            "align": "none",
            "composite": "local_person_white_bg",
        },
        "width": int(edited.shape[1]),
        "height": int(edited.shape[0]),
    }


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


def _edit_openrouter(
    data: bytes,
    mime: str,
    src: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generative white-bg via OpenRouter + original-face protect."""
    src_p = prepare_for_cutout(src)
    src_p = _resize_max_side(src_p, config.MAX_IMAGE_SIDE)

    raw = edit_selfie(data, mime=mime)
    decoded = _decode_any(raw)
    if decoded is None:
        raise RuntimeError("Edited image decode failed")
    edited = composite_on_white(decoded)

    out, protected, fp_meta = apply_face_protect(src_p, edited)
    out = force_white_background(out, tol=48)
    out = _gentle_light_outside_face(out)
    out = force_white_background(out, tol=45)

    return out, {
        "model": config.OPENROUTER_IMAGE_MODEL,
        "cutout": "openrouter",
        "face_protected": bool(protected),
        "face_protect": {
            **fp_meta,
            "composite": "openrouter_bg_original_face",
        },
        "face_source": "original",
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    }


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


def run_edit_stage(
    data: bytes,
    mime: str = "image/jpeg",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Stage 1 entry: bytes → edited BGR (white bg, not cropped).

    Strategy:
      EDIT_BACKEND=openrouter|auto → generative OpenRouter, local cutout fallback.
      EDIT_BACKEND=local → ONNX/MediaPipe cutout only.
    """
    backend = config.EDIT_BACKEND
    has_or_key = bool(config.OPENROUTER_API_KEY)
    prefer_or = backend in ("openrouter", "auto") and has_or_key
    prefer_local = backend in ("local", "") or (backend == "auto" and not has_or_key)
    allow_or = has_or_key and backend in ("openrouter", "auto")
    allow_local = backend in ("local", "auto", "openrouter", "")

    if backend not in ("local", "openrouter", "auto", ""):
        prefer_or = has_or_key
        prefer_local = not prefer_or
    elif backend == "openrouter":
        prefer_or = has_or_key
        prefer_local = True
    elif backend == "auto":
        prefer_or = has_or_key
        prefer_local = True
    elif backend in ("local", ""):
        prefer_or = False
        prefer_local = True

    meta: dict[str, Any] = {"stage": "edit"}
    local_err: Exception | None = None
    or_err: Exception | None = None

    src = _decode_image(data)
    if src is None:
        raise RuntimeError("decode_error")

    if prefer_or and allow_or:
        try:
            edited, or_meta = _edit_openrouter(data, mime, src)
            meta.update(or_meta)
            return edited, meta
        except Exception as e:
            or_err = e
            log.warning("OpenRouter edit failed, falling back to local: %s", e)

    if prefer_local and allow_local:
        try:
            edited, local_meta = edit_selfie_local(src)
            meta.update(local_meta)
            meta["model"] = local_meta.get("cutout", "mediapipe")
            return edited, meta
        except Exception as e:
            local_err = e
            log.warning("Local edit failed: %s", e)
            if backend == "local" or not allow_or:
                raise

    if allow_or and prefer_local:
        try:
            edited, or_meta = _edit_openrouter(data, mime, src)
            meta.update(or_meta)
            if local_err is not None:
                meta["local_fallback"] = str(local_err)[:200]
            return edited, meta
        except Exception as e:
            or_err = e
            log.warning("OpenRouter fallback failed: %s", e)

    if prefer_or and allow_local:
        try:
            edited, local_meta = edit_selfie_local(src)
            meta.update(local_meta)
            meta["model"] = local_meta.get("cutout", "mediapipe")
            if or_err is not None:
                meta["openrouter_fallback"] = str(or_err)[:200]
            return edited, meta
        except Exception:
            if or_err:
                raise or_err
            raise

    if or_err:
        raise or_err
    if local_err:
        raise local_err
    raise RuntimeError(f"No edit backend available (EDIT_BACKEND={backend})")
