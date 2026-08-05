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
    blur = _blur_score(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker().detect(mp_image)

    face_count = len(result.face_landmarks) if result.face_landmarks else 0
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

    if (
        result.facial_transformation_matrixes
        and len(result.facial_transformation_matrixes) > 0
    ):
        mat = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
        yaw, pitch, roll = _euler_from_matrix(mat)
    else:
        yaw, pitch, roll = _pose_from_landmarks(result.face_landmarks[0])

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
