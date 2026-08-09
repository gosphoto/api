"""Step 1 — edit selfie: normalize + white studio background.

No passport crop here. Output is a cleaned full-frame portrait ready for /api/crop.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np

from . import config
from .bg import last_cutout_backend, prepare_for_cutout, white_background_local
from .compose_bg import composite_on_white
from .face_protect import align_edit_to_original, face_protect_mask
from .gate import _decode_image, _resize_max_side
from .openrouter import edit_selfie, edit_selfie_riverflow
from .whitening import force_white_background

log = logging.getLogger("gosphoto-gate")


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


def run_edit_riverflow(
    data: bytes,
    mime: str = "image/jpeg",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Riverflow v2.5 Pro: solid #FFFFFF bg via native background_mode."""
    src = _decode_image(data)
    if src is None:
        raise RuntimeError("decode_error")
    src_p = prepare_for_cutout(src)
    src_p = _resize_max_side(src_p, config.MAX_IMAGE_SIDE)

    ok, buf = cv2.imencode(".jpg", src_p, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("encode_for_riverflow_failed")
    orig_jpg = buf.tobytes()

    raw = edit_selfie_riverflow(orig_jpg, mime="image/jpeg")
    decoded = _decode_any(raw)
    if decoded is None:
        raise RuntimeError("Riverflow decode failed")
    out = composite_on_white(decoded)
    out = _uniform_max_side(out, config.MAX_IMAGE_SIDE)
    out = force_white_background(out, tol=48)

    bg_mode = config.RIVERFLOW_BG_MODE or "solid"
    return out, {
        "model": config.RIVERFLOW_MODEL,
        "cutout": "riverflow",
        "background_mode": bg_mode,
        "background_hex_color": (
            config.RIVERFLOW_BG_HEX if bg_mode == "solid" else None
        ),
        "image_size": config.RIVERFLOW_IMAGE_SIZE,
        "reasoning": config.RIVERFLOW_REASONING,
        "face_protected": False,
        "passes": 1,
        "prompt": "gosuslugi_riverflow",
        "src_size": [int(src_p.shape[1]), int(src_p.shape[0])],
        "kept_model_aspect": True,
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    }


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


def _person_alpha_on_white(local_bgr: np.ndarray) -> np.ndarray:
    """Soft 0..1 person matte from a white-bg cutout (no landmark warp)."""
    gray = cv2.cvtColor(local_bgr, cv2.COLOR_BGR2GRAY)
    f = local_bgr.astype(np.float32)
    chroma = np.linalg.norm(f - f.mean(axis=2, keepdims=True), axis=2)
    hard = ((gray < 248) | (chroma > 10)).astype(np.uint8) * 255
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (hard > 0).astype(np.uint8), connectivity=8
    )
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        hard = ((labels == largest).astype(np.uint8)) * 255
    # Slight erode so silueta fringe/bg crumbs don't stick
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(hard, (0, 0), 1.2).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def _edit_openrouter(
    data: bytes,
    mime: str,
    src: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Identity-safe white bg: full local person, never paste/warp a face layer.

    OpenRouter redraws the head in a different place in-frame; compositing it
    under/over the local face always looks 'съехало'. Keep OR available for
    labs via EDIT_USE_OR_PIXELS=1, default off.
    """
    src_p = prepare_for_cutout(src)
    src_p = _resize_max_side(src_p, config.MAX_IMAGE_SIDE)
    local = white_background_local(src_p)

    use_or_pixels = os.getenv("EDIT_USE_OR_PIXELS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    or_meta: dict[str, Any] = {"or_pixels": False}
    if use_or_pixels:
        raw = edit_selfie(data, mime=mime)
        decoded = _decode_any(raw)
        if decoded is None:
            raise RuntimeError("Edited image decode failed")
        edited = composite_on_white(decoded)
        aligned, did_align = align_edit_to_original(local, edited)
        person = _person_alpha_on_white(local)
        # Hard matte — no soft OR ghost at edges
        hard = (person >= 0.55).astype(np.float32)
        hard = cv2.GaussianBlur(hard, (0, 0), 0.6)
        a = hard[:, :, None]
        white = np.full_like(local, 255, dtype=np.float32)
        # Background = white (not OR person ghosts); person = local only
        out = (white * (1.0 - a) + local.astype(np.float32) * a).astype(np.uint8)
        or_meta = {
            "or_pixels": True,
            "or_aligned": did_align,
            "note": "OR aligned but discarded outside hard local matte",
        }
    else:
        # Pure local person on white — face cannot shift relative to head
        person = _person_alpha_on_white(local)
        hard = (person >= 0.45).astype(np.float32)
        hard = cv2.GaussianBlur(hard, (0, 0), 0.8)
        a = hard[:, :, None]
        white = np.full_like(local, 255, dtype=np.float32)
        out = (white * (1.0 - a) + local.astype(np.float32) * a).astype(np.uint8)

    out = force_white_background(out, tol=48)
    return out, {
        "model": config.OPENROUTER_IMAGE_MODEL if use_or_pixels else last_cutout_backend,
        "cutout": "openrouter+local" if use_or_pixels else last_cutout_backend,
        "face_protected": True,
        "face_protect": {
            "model": "local_person_matte",
            "applied": True,
            "align": "none",
            "composite": "local_person_white_bg",
            **or_meta,
        },
        "face_source": last_cutout_backend,
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    }


def prepare_skip_edit(data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    """Crop-only path: light bleach on already-studio input, no Riverflow."""
    src = _decode_image(data)
    if src is None:
        raise RuntimeError("decode_error")
    src_p = prepare_for_cutout(src)
    src_p = _resize_max_side(src_p, config.MAX_IMAGE_SIDE)
    out = force_white_background(src_p, tol=40)
    return out, {
        "stage": "edit",
        "model": None,
        "cutout": "skip_edit",
        "skipped_riverflow": True,
        "background_mode": "passthrough",
        "face_protected": True,
        "passes": 0,
        "src_size": [int(src_p.shape[1]), int(src_p.shape[0])],
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
    }


def run_edit_stage(
    data: bytes,
    mime: str = "image/jpeg",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Stage 1 entry: bytes → edited BGR (white bg, not cropped).

    Default: Riverflow v2.5 Pro (solid #FFFFFF). On failure → local cutout.
    EDIT_BACKEND=local forces silueta/ONNX path only.
    """
    backend = (config.EDIT_BACKEND or "riverflow").strip().lower()
    has_or_key = bool(config.OPENROUTER_API_KEY)
    use_riverflow = backend in ("riverflow", "openrouter", "auto") and has_or_key
    allow_local = backend in ("local", "auto", "riverflow", "openrouter", "")

    meta: dict[str, Any] = {"stage": "edit"}
    river_err: Exception | None = None

    src = _decode_image(data)
    if src is None:
        raise RuntimeError("decode_error")

    if use_riverflow:
        try:
            edited, rf_meta = run_edit_riverflow(data, mime=mime)
            meta.update(rf_meta)
            return edited, meta
        except Exception as e:
            river_err = e
            log.warning("Riverflow edit failed, falling back to local: %s", e)
            if backend == "riverflow" and not allow_local:
                raise

    if allow_local:
        try:
            edited, local_meta = edit_selfie_local(src)
            meta.update(local_meta)
            meta["model"] = local_meta.get("cutout", "mediapipe")
            if river_err is not None:
                meta["riverflow_fallback"] = str(river_err)[:200]
            return edited, meta
        except Exception as e:
            if river_err:
                raise river_err
            raise e

    if river_err:
        raise river_err
    raise RuntimeError(f"No edit backend available (EDIT_BACKEND={backend})")
