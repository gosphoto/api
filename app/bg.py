"""Local person cutout → solid white background (no paid API).

Default: ONNX silueta + morph-close silhouette (lab winner 2026-08-06).
Fallbacks: u2netp → MediaPipe. Optional rembg.
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

_ONNX_SIZE = 320

last_cutout_backend: str = "silueta"


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


@lru_cache(maxsize=4)
def _onnx_session(path_str: str):
    import onnxruntime as ort

    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX cutout model missing: {path}")
    log.info("Loading ONNX cutout from %s", path)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _onnx_model_path(name: str) -> Path:
    name = name.lower()
    if name == "silueta":
        return Path(config.SILUETA_MODEL_PATH)
    if name == "u2net":
        return Path(config.U2NET_MODEL_PATH)
    return Path(config.U2NETP_MODEL_PATH)


def warmup_cutout() -> None:
    backend = (config.EDIT_CUTOUT or "silueta").strip().lower()
    if backend == "rembg":
        _rembg_session()
        return
    if backend == "mediapipe":
        _segmenter()
        return
    for cand in (backend, "silueta", "u2netp"):
        try:
            _onnx_session(str(_onnx_model_path(cand)))
            return
        except Exception as e:
            log.warning("cutout warmup %s failed: %s", cand, e)
    _segmenter()


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask.astype(np.float32)
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)


def _largest_component(m8: np.ndarray) -> np.ndarray:
    hard = (m8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    if num <= 1:
        return m8
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (labels == largest).astype(np.uint8) * 255
    return keep


def _face_lock_mask(bgr: np.ndarray) -> np.ndarray:
    """Dilated face/hair/upper-torso lock so cutout never eats the face."""
    from .face_restore import _face_mask, _landmarks_xy

    lm = _landmarks_xy(bgr)
    if lm is None:
        return np.zeros(bgr.shape[:2], np.uint8)
    m = (_face_mask(bgr.shape[:2], lm) * 255.0).astype(np.uint8)
    m = cv2.dilate(m, np.ones((31, 31), np.uint8), iterations=2)
    return m


def _onnx_confidence(bgr: np.ndarray, model: str) -> np.ndarray:
    path = _onnx_model_path(model)
    sess = _onnx_session(str(path))
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = cv2.resize(rgb, (_ONNX_SIZE, _ONNX_SIZE)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = (im - mean) / std
    tensor = im.transpose(2, 0, 1)[None, ...].astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    pred = out[0][0] if out.ndim == 4 else out[0]
    pred = pred.astype(np.float32)
    pred = (pred - pred.min()) / (float(pred.max() - pred.min()) + 1e-8)
    return _resize_mask(pred, (w, h))


def _silhouette_alpha(
    conf: np.ndarray,
    face_lock: np.ndarray,
    *,
    thr: float = 0.52,
    close_k: int = 41,
    erode: int = 2,
    feather: float = 1.1,
) -> np.ndarray:
    """Hard person matte with morph-close to seal neck/shoulder caves."""
    m = (conf >= thr).astype(np.uint8) * 255
    if face_lock is not None and face_lock.any():
        m = cv2.bitwise_or(m, face_lock)
    m = _largest_component(m)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if erode > 0:
        m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=erode)
        if face_lock is not None and face_lock.any():
            m = cv2.bitwise_or(m, cv2.erode(face_lock, np.ones((7, 7), np.uint8)))
    a = cv2.GaussianBlur(m, (0, 0), feather).astype(np.float32) / 255.0
    return np.clip(a, 0.0, 1.0)


def _composite_on_white(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = np.clip(mask.astype(np.float32), 0.0, 1.0)[:, :, None]
    out = bgr.astype(np.float32) * a + 255.0 * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _white_bg_onnx(bgr: np.ndarray, model: str) -> np.ndarray:
    bgr = prepare_for_cutout(bgr)
    conf = _onnx_confidence(bgr, model)
    face = _face_lock_mask(bgr)
    alpha = _silhouette_alpha(conf, face)
    return _composite_on_white(bgr, alpha)


def _white_bg_mediapipe(bgr: np.ndarray) -> np.ndarray:
    bgr = prepare_for_cutout(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _segmenter().segment(mp_image)
    if not result.confidence_masks:
        raise RuntimeError("selfie_segmenter_no_mask")
    conf = _resize_mask(
        result.confidence_masks[0].numpy_view(), (bgr.shape[1], bgr.shape[0])
    )
    face = _face_lock_mask(bgr)
    alpha = _silhouette_alpha(conf, face, thr=0.35, close_k=35, erode=1)
    return _composite_on_white(bgr, alpha)


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
    face = _face_lock_mask(rgb)
    a = _silhouette_alpha(alpha.astype(np.float32) / 255.0, face, thr=0.45)
    return _composite_on_white(rgb, a)


def white_background_local(bgr: np.ndarray) -> np.ndarray:
    """Replace background with #FFFFFF. Sets last_cutout_backend."""
    global last_cutout_backend
    backend = (config.EDIT_CUTOUT or "silueta").strip().lower()

    if backend == "rembg":
        try:
            out = _white_bg_rembg(bgr)
            last_cutout_backend = "rembg"
            return out
        except Exception as e:
            log.warning("rembg failed, falling back: %s", e)

    if backend == "mediapipe":
        last_cutout_backend = "mediapipe"
        return _white_bg_mediapipe(bgr)

    # Preferred ONNX chain
    chain: list[str]
    if backend in ("silueta", "u2netp", "u2net"):
        chain = [backend, "silueta", "u2netp"]
    else:  # auto / unknown
        chain = ["silueta", "u2netp"]

    seen: set[str] = set()
    for cand in chain:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            out = _white_bg_onnx(bgr, cand)
            last_cutout_backend = cand
            return out
        except Exception as e:
            log.warning("%s cutout failed: %s", cand, e)

    last_cutout_backend = "mediapipe"
    return _white_bg_mediapipe(bgr)


def warmup_rembg() -> None:
    warmup_cutout()
