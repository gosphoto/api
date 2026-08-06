"""Local person cutout → solid white background (no paid API).

Prefer ONNX u2netp (sharp edges, VPS-safe). Fall back to MediaPipe.
Optional: rembg if EDIT_CUTOUT=rembg.
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

_U2NETP_SIZE = 320


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


@lru_cache(maxsize=1)
def _u2netp_session():
    import onnxruntime as ort

    path = Path(config.U2NETP_MODEL_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"u2netp model missing: {path}")
    log.info("Loading u2netp ONNX from %s", path)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def warmup_cutout() -> None:
    backend = config.EDIT_CUTOUT
    if backend == "rembg":
        _rembg_session()
    elif backend in ("u2netp", "auto", ""):
        try:
            _u2netp_session()
            return
        except Exception as e:
            log.warning("u2netp warmup failed, will use mediapipe: %s", e)
            _segmenter()
    else:
        _segmenter()


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (w, h)."""
    w, h = size
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask.astype(np.float32)
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def _largest_component_soft(m8: np.ndarray) -> np.ndarray:
    """Keep largest blob; preserve soft edges near it."""
    hard = (m8 > 120).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if num <= 1:
        return m8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (labels == largest).astype(np.uint8)
    keep = cv2.dilate(keep, np.ones((11, 11), np.uint8), iterations=1)
    return cv2.bitwise_and(m8, keep * 255)


def _soft_matte_from_confidence(mask: np.ndarray) -> np.ndarray:
    """Turn a confidence map into a clean soft alpha."""
    mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
    mask = np.clip((mask - 0.15) / 0.55, 0.0, 1.0)
    m8 = (mask * 255.0).astype(np.uint8)
    m8 = cv2.bilateralFilter(m8, d=7, sigmaColor=40, sigmaSpace=7)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    m8 = _largest_component_soft(m8)
    # Inward bias then feather — kills colour fringe without stair-steps
    m8 = cv2.erode(m8, np.ones((3, 3), np.uint8), iterations=1)
    m8 = cv2.GaussianBlur(m8, (0, 0), sigmaX=1.2)
    a = np.clip(m8.astype(np.float32) / 255.0, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def _composite_on_white(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Alpha over #FFFFFF with mid-tone decontamination (anti colour fringe)."""
    a = np.clip(mask.astype(np.float32), 0.0, 1.0)
    a3 = a[:, :, None]
    fg = bgr.astype(np.float32)
    white = np.full_like(fg, 255.0)
    soft = ((a > 0.04) & (a < 0.92)).astype(np.float32)[:, :, None]
    fg = fg * (1.0 - 0.60 * soft) + white * (0.60 * soft)
    out = fg * a3 + white * (1.0 - a3)
    return np.clip(out, 0, 255).astype(np.uint8)


def _u2netp_mask(bgr: np.ndarray) -> np.ndarray:
    sess = _u2netp_session()
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = cv2.resize(rgb, (_U2NETP_SIZE, _U2NETP_SIZE)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = (im - mean) / std
    tensor = im.transpose(2, 0, 1)[None, ...].astype(np.float32)
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: tensor})[0]
    pred = out[0][0] if out.ndim == 4 else out[0]
    pred = pred.astype(np.float32)
    pred = (pred - pred.min()) / (float(pred.max() - pred.min()) + 1e-8)
    return _resize_mask(pred, (w, h))


def _white_bg_u2netp(bgr: np.ndarray) -> np.ndarray:
    bgr = prepare_for_cutout(bgr)
    mask = _soft_matte_from_confidence(_u2netp_mask(bgr))
    return _composite_on_white(bgr, mask)


def _white_bg_mediapipe(bgr: np.ndarray) -> np.ndarray:
    bgr = prepare_for_cutout(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _segmenter().segment(mp_image)
    if not result.confidence_masks:
        raise RuntimeError("selfie_segmenter_no_mask")
    raw = _resize_mask(result.confidence_masks[0].numpy_view(), (bgr.shape[1], bgr.shape[0]))
    # Optional GrabCut tighten for MP only (u2netp already sharp)
    gc = _grabcut_refine(bgr, np.clip((raw - 0.18) / 0.52, 0, 1))
    blended = np.maximum(gc * 0.75, raw * 0.55)
    mask = _soft_matte_from_confidence(blended)
    return _composite_on_white(bgr, mask)


def _grabcut_refine(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[mask >= 0.80] = cv2.GC_FGD
    gc[(mask >= 0.35) & (mask < 0.80)] = cv2.GC_PR_FGD
    gc[(mask >= 0.12) & (mask < 0.35)] = cv2.GC_PR_BGD
    if int((gc == cv2.GC_FGD).sum()) < 64 or int((gc == cv2.GC_BGD).sum()) < 64:
        return mask
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
    except cv2.error as e:
        log.warning("grabCut refine skipped: %s", e)
        return mask
    return np.where(
        (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 1.0, 0.0
    ).astype(np.float32)


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
    a = _soft_matte_from_confidence(alpha.astype(np.float32) / 255.0)
    return _composite_on_white(rgb, a)


last_cutout_backend: str = "u2netp"


def white_background_local(bgr: np.ndarray) -> np.ndarray:
    """Replace background with #FFFFFF. Sets last_cutout_backend for metadata."""
    global last_cutout_backend
    backend = (config.EDIT_CUTOUT or "u2netp").strip().lower()

    if backend == "rembg":
        try:
            out = _white_bg_rembg(bgr)
            last_cutout_backend = "rembg"
            return out
        except Exception as e:
            log.warning("rembg failed, falling back: %s", e)

    if backend in ("u2netp", "auto", "", "rembg"):
        try:
            out = _white_bg_u2netp(bgr)
            last_cutout_backend = "u2netp"
            return out
        except Exception as e:
            log.warning("u2netp failed, falling back to mediapipe: %s", e)

    last_cutout_backend = "mediapipe"
    return _white_bg_mediapipe(bgr)


# Back-compat aliases used by main.py
def warmup_rembg() -> None:
    warmup_cutout()
