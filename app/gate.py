"""Passport photo input gate: MediaPipe Face Landmarker + OpenCV blur."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from . import config

# MediaPipe Face Mesh indices
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_NOSE_TIP = 1
_CHIN = 152
_FOREHEAD = 10


@dataclass
class GateResult:
    ok: bool
    reason: str | None
    message: str
    face_count: int
    metrics: dict[str, Any]


_MESSAGES = {
    "ok": "Фото подходит для обработки",
    "no_face": "Лицо не найдено — загрузите селфи анфас",
    "multiple_faces": "На фото должно быть одно лицо",
    "pose_yaw": "Смотрите прямо в камеру (слишком сильный поворот головы)",
    "pose_pitch": "Держите голову ровно (не задирайте и не опускайте подбородок)",
    "pose_roll": "Выровняйте голову (не наклоняйте вбок)",
    "blur": "Фото размыто — переснимите при хорошем свете",
    "decode_error": "Не удалось прочитать изображение",
}


@lru_cache(maxsize=1)
def _landmarker() -> vision.FaceLandmarker:
    if not config.MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=3,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def warmup() -> None:
    """Load model once at process start."""
    _landmarker()


def _decode_image(data: bytes) -> np.ndarray | None:
    """Decode image bytes with EXIF orientation applied (phone selfies)."""
    from io import BytesIO

    from PIL import Image, ImageOps

    try:
        pil = Image.open(BytesIO(data))
        pil = ImageOps.exif_transpose(pil)
        if pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        elif pil.mode == "L":
            pil = pil.convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return bgr
    scale = max_side / side
    return cv2.resize(
        bgr,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _blur_score(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _euler_from_matrix(mat4: np.ndarray) -> tuple[float, float, float]:
    """Extract yaw/pitch/roll (degrees) from 4x4 facial transformation matrix."""
    r = mat4[:3, :3]
    # ZYX-ish from rotation matrix (MediaPipe camera/face frame)
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(r[1, 0], r[0, 0])
        roll = math.atan2(r[2, 1], r[2, 2])
    else:
        pitch = math.atan2(-r[2, 0], sy)
        yaw = math.atan2(-r[0, 1], r[1, 1])
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _pose_from_landmarks(landmarks) -> tuple[float, float, float]:
    """Fallback pose estimate from a few mesh points (degrees)."""
    le = landmarks[_LEFT_EYE_OUTER]
    re = landmarks[_RIGHT_EYE_OUTER]
    nose = landmarks[_NOSE_TIP]
    chin = landmarks[_CHIN]
    forehead = landmarks[_FOREHEAD]

    dx = re.x - le.x
    dy = re.y - le.y
    roll = math.degrees(math.atan2(dy, dx))

    mid_x = (le.x + re.x) / 2
    eye_dist = math.hypot(dx, dy) or 1e-6
    yaw = math.degrees(math.atan2((nose.x - mid_x) / eye_dist, 1.0)) * 1.5

    face_h = abs(chin.y - forehead.y) or 1e-6
    mid_y = (le.y + re.y) / 2
    pitch = math.degrees(math.atan2((nose.y - mid_y) / face_h, 1.0)) * 1.5

    return yaw, pitch, roll


def _encode_jpeg(bgr: np.ndarray, quality: int = 92) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def _analyze_bgr(
    bgr: np.ndarray,
) -> tuple[int, float, float, float, float, bool, float]:
    """face_count, yaw, pitch, roll, blur, chin_below_forehead, face_center_y."""
    blur = _blur_score(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker().detect(mp_image)
    face_count = len(result.face_landmarks) if result.face_landmarks else 0
    if face_count != 1:
        return face_count, 0.0, 0.0, 0.0, blur, False, 0.5

    landmarks = result.face_landmarks[0]
    if (
        result.facial_transformation_matrixes
        and len(result.facial_transformation_matrixes) > 0
    ):
        mat = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
        yaw, pitch, roll = _euler_from_matrix(mat)
    else:
        yaw, pitch, roll = _pose_from_landmarks(landmarks)

    chin = landmarks[_CHIN]
    forehead = landmarks[_FOREHEAD]
    upright = chin.y > forehead.y
    face_center_y = float((chin.y + forehead.y) / 2.0)
    return face_count, yaw, pitch, roll, blur, upright, face_center_y


def _pose_score(
    yaw: float,
    pitch: float,
    roll: float,
    upright: bool,
    face_center_y: float = 0.5,
) -> float:
    """Lower is better. Penalize upside-down / low-in-frame faces.

    MediaPipe often paints a canonical upright mesh onto an inverted head near
    the bottom of the frame; chin_below_forehead alone cannot catch that, so we
    also prefer faces higher in the image (typical portrait framing).
    """
    score = abs(yaw) + abs(pitch) + abs(roll)
    if not upright:
        score += 180.0
    score += 50.0 * face_center_y
    return score


def upright_image(data: bytes) -> tuple[bytes | None, dict[str, Any]]:
    """
    Pick the best 0/90/180/270 orientation (sideways / inverted phone photos).

    Returns (jpeg_bytes or None, meta). Meta always includes rotation_deg.
    """
    src = _decode_image(data)
    if src is None:
        return None, {"rotation_deg": 0}

    src = _resize_max_side(src, config.MAX_IMAGE_SIDE)
    best_k = 0
    best_score = float("inf")
    best_bgr = src
    best_stats: tuple[int, float, float, float, float, bool, float] | None = None

    for k in (0, 1, 2, 3):
        cand = src if k == 0 else np.ascontiguousarray(np.rot90(src, k))
        face_count, yaw, pitch, roll, blur, upright, face_cy = _analyze_bgr(cand)
        if face_count != 1:
            continue
        score = _pose_score(yaw, pitch, roll, upright, face_cy)
        if score < best_score:
            best_score = score
            best_k = k
            best_bgr = cand
            best_stats = (face_count, yaw, pitch, roll, blur, upright, face_cy)

    meta: dict[str, Any] = {"rotation_deg": int(best_k * 90)}
    if best_stats is not None:
        _, yaw, pitch, roll, blur, upright, face_cy = best_stats
        meta.update(
            {
                "upright_yaw": round(yaw, 2),
                "upright_pitch": round(pitch, 2),
                "upright_roll": round(roll, 2),
                "upright_blur": round(blur, 2),
                "chin_below_forehead": upright,
                "face_center_y": round(face_cy, 3),
            }
        )

    if best_k == 0:
        # Keep original bytes when no rotation needed (preserve quality).
        return data, meta

    encoded = _encode_jpeg(best_bgr)
    return encoded, meta


def validate_image(data: bytes) -> GateResult:
    bgr = _decode_image(data)
    if bgr is None:
        return GateResult(
            ok=False,
            reason="decode_error",
            message=_MESSAGES["decode_error"],
            face_count=0,
            metrics={},
        )

    bgr = _resize_max_side(bgr, config.MAX_IMAGE_SIDE)
    face_count, yaw, pitch, roll, blur, _upright, _face_cy = _analyze_bgr(bgr)
    metrics: dict[str, Any] = {
        "blur": round(blur, 2),
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
        "face_count": face_count,
    }

    if face_count == 0:
        return GateResult(False, "no_face", _MESSAGES["no_face"], 0, metrics)
    if face_count > 1:
        return GateResult(
            False, "multiple_faces", _MESSAGES["multiple_faces"], face_count, metrics
        )

    metrics.update(
        {
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "roll": round(roll, 2),
        }
    )

    if abs(yaw) > config.MAX_YAW_DEG:
        return GateResult(False, "pose_yaw", _MESSAGES["pose_yaw"], 1, metrics)
    if abs(pitch) > config.MAX_PITCH_DEG:
        return GateResult(False, "pose_pitch", _MESSAGES["pose_pitch"], 1, metrics)
    if abs(roll) > config.MAX_ROLL_DEG:
        return GateResult(False, "pose_roll", _MESSAGES["pose_roll"], 1, metrics)
    if blur < config.MIN_BLUR_VARIANCE:
        return GateResult(False, "blur", _MESSAGES["blur"], 1, metrics)

    return GateResult(True, None, _MESSAGES["ok"], 1, metrics)


def prepare_upload(data: bytes) -> tuple[bytes, GateResult]:
    """
    Auto-upright sideways uploads, then run gate on the oriented frame.

    Downstream (Riverflow) must use the returned bytes.
    """
    oriented, upright_meta = upright_image(data)
    if oriented is None:
        return data, GateResult(
            False,
            "decode_error",
            _MESSAGES["decode_error"],
            0,
            upright_meta,
        )
    gate = validate_image(oriented)
    gate.metrics = {**gate.metrics, **upright_meta}
    return oriented, gate
