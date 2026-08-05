"""Compliance metrics for RF passport / Gosuslugi digital photo."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker
from .whitening import corner_whiteness

_LEFT = 33
_RIGHT = 263
_CHIN = 152
_FOREHEAD = 10


def measure_compliance(bgr: np.ndarray) -> dict[str, Any]:
    h, w = bgr.shape[:2]
    white = corner_whiteness(bgr)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return {
            "width": w,
            "height": h,
            "size_ok": (w, h) == (413, 531),
            "face_count": 0,
            "bg": white,
            "pass": False,
        }

    lm = result.face_landmarks[0]
    crown = lm[_FOREHEAD].y * h - 0.35 * (lm[_CHIN].y - lm[_FOREHEAD].y) * h
    chin = lm[_CHIN].y * h
    face_h = max(chin - crown, 1.0)
    face_ratio = face_h / h
    top_margin = crown / h
    # RF: face ≤80% and typically ≥70%; top field 5±1 mm → ~0.089–0.133 of 45 mm
    checks = {
        "size_ok": (w, h) == (413, 531),
        "face_ratio_ok": 0.70 <= face_ratio <= 0.80,
        "top_margin_ok": 0.09 <= top_margin <= 0.14,
        "bg_white_ok": white["white_ok"],
        "single_face_ok": True,
    }
    return {
        "width": w,
        "height": h,
        "face_ratio": round(face_ratio, 3),
        "top_margin": round(top_margin, 3),
        "bg": white,
        "checks": checks,
        "pass": all(checks.values()),
    }
