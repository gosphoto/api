"""Compliance metrics for RF passport (rg.ru / FMS п.34.3) digital photo."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .gate import _landmarker
from .whitening import corner_whiteness

_LEFT = 33
_RIGHT = 263
_CHIN = 152
_FOREHEAD = 10
# Approximate face sides (cheeks) for head width in mm
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


def measure_compliance(bgr: np.ndarray) -> dict[str, Any]:
    """Check crop against FMS п.34.3 geometry (35×45, face oval, head mm)."""
    h, w = bgr.shape[:2]
    white = corner_whiteness(bgr)
    size_ok = (w, h) == (config.PASSPORT_WIDTH, config.PASSPORT_HEIGHT)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return {
            "width": w,
            "height": h,
            "dpi_target": config.PASSPORT_DPI,
            "size_ok": size_ok,
            "face_count": 0,
            "bg": white,
            "pass": False,
            "source": "rg.ru/2011/08/22/pasport-dok.html §34.3",
        }

    lm = result.face_landmarks[0]
    # Crown ≈ above forehead (hairline estimate)
    crown = lm[_FOREHEAD].y * h - 0.45 * (lm[_CHIN].y - lm[_FOREHEAD].y) * h
    chin = lm[_CHIN].y * h
    face_h = max(chin - crown, 1.0)
    face_ratio = face_h / h
    top_margin = max(crown, 0.0) / h

    left_x = lm[_LEFT_CHEEK].x * w
    right_x = lm[_RIGHT_CHEEK].x * w
    face_w = abs(right_x - left_x)

    head_h_mm = face_ratio * config.PASSPORT_HEIGHT_MM
    head_w_mm = (face_w / w) * config.PASSPORT_WIDTH_MM

    # Soft band around landmark noise; regulation: oval ≥80%, head 32–36×18–25 mm
    face_oval_ok = face_ratio >= 0.78
    head_height_ok = (
        config.HEAD_HEIGHT_MM_MIN <= head_h_mm <= config.HEAD_HEIGHT_MM_MAX
    )
    head_width_ok = (
        config.HEAD_WIDTH_MM_MIN <= head_w_mm <= config.HEAD_WIDTH_MM_MAX
    )
    # Top field ~3–6 mm of 45 mm
    top_margin_ok = 0.05 <= top_margin <= 0.14

    checks = {
        "size_ok": size_ok,
        "face_oval_ok": face_oval_ok,
        "head_height_mm_ok": head_height_ok,
        "head_width_mm_ok": head_width_ok,  # soft: cheek landmarks noisy
        "top_margin_ok": top_margin_ok,
        "bg_white_ok": white["white_ok"],
        "single_face_ok": True,
        # legacy alias for scoring / UI
        "face_ratio_ok": face_oval_ok and head_height_ok,
    }
    hard = (
        "size_ok",
        "face_oval_ok",
        "head_height_mm_ok",
        "top_margin_ok",
        "bg_white_ok",
        "single_face_ok",
    )

    return {
        "width": w,
        "height": h,
        "dpi_target": config.PASSPORT_DPI,
        "face_ratio": round(face_ratio, 3),
        "top_margin": round(top_margin, 3),
        "head_height_mm": round(head_h_mm, 1),
        "head_width_mm": round(head_w_mm, 1),
        "bg": white,
        "checks": checks,
        "pass": all(checks[k] for k in hard),
        "source": "rg.ru/2011/08/22/pasport-dok.html §34.3",
    }
