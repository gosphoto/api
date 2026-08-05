"""Local passport crop: MediaPipe geometry → 35×45 @ 300dpi."""

from __future__ import annotations

import math
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .gate import _landmarker  # reuse loaded model

_LEFT_EYE = 33
_RIGHT_EYE = 263
_CHIN = 152
_FOREHEAD = 10


def crop_passport(bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Align/roll-correct and crop to passport canvas. Returns BGR + metrics."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker().detect(mp_image)
    if not result.face_landmarks:
        raise ValueError("no_face_after_edit")

    lm = result.face_landmarks[0]
    h, w = bgr.shape[:2]

    le = lm[_LEFT_EYE]
    re = lm[_RIGHT_EYE]
    chin = lm[_CHIN]
    forehead = lm[_FOREHEAD]

    # Roll from eye line
    dx = (re.x - le.x) * w
    dy = (re.y - le.y) * h
    roll_deg = math.degrees(math.atan2(dy, dx))

    center = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(center, roll_deg, 1.0)
    rotated = cv2.warpAffine(
        bgr,
        rot,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    # Re-detect on rotated for accurate crop box
    rgb2 = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
    mp2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb2)
    result2 = _landmarker().detect(mp2)
    if not result2.face_landmarks:
        rotated_src = rotated
        lm2 = lm
        # map approx with same normalized coords (rough fallback)
    else:
        rotated_src = rotated
        lm2 = result2.face_landmarks[0]

    le = lm2[_LEFT_EYE]
    re = lm2[_RIGHT_EYE]
    chin = lm2[_CHIN]
    forehead = lm2[_FOREHEAD]

    eye_y = ((le.y + re.y) / 2) * h
    chin_y = chin.y * h
    forehead_y = forehead.y * h
    # crown ≈ above forehead
    crown_y = forehead_y - 0.35 * (chin_y - forehead_y)
    face_h = max(chin_y - crown_y, 1.0)
    mid_x = ((le.x + re.x) / 2) * w

    out_w = config.PASSPORT_WIDTH
    out_h = config.PASSPORT_HEIGHT
    target_face = config.PASSPORT_FACE_RATIO * out_h
    scale = target_face / face_h

    # crop window in source coords
    crop_h = out_h / scale
    crop_w = out_w / scale
    top = crown_y - config.PASSPORT_TOP_MARGIN * crop_h
    left = mid_x - crop_w / 2

    # pad source with white so crop can go outside
    pad = int(max(crop_w, crop_h))
    canvas = cv2.copyMakeBorder(
        rotated_src,
        pad,
        pad,
        pad,
        pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    left_p = left + pad
    top_p = top + pad
    x0 = int(round(left_p))
    y0 = int(round(top_p))
    x1 = int(round(left_p + crop_w))
    y1 = int(round(top_p + crop_h))

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(canvas.shape[1], x1)
    y1 = min(canvas.shape[0], y1)
    patch = canvas[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("empty_crop")

    out = cv2.resize(patch, (out_w, out_h), interpolation=cv2.INTER_AREA)
    metrics = {
        "roll_corrected_deg": round(roll_deg, 2),
        "width": out_w,
        "height": out_h,
        "face_ratio_target": config.PASSPORT_FACE_RATIO,
    }
    return out, metrics


def encode_jpeg(bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("jpeg_encode_failed")
    return buf.tobytes()
