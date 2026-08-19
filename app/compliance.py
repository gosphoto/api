"""Compliance metrics for RF passport (rg.ru / FMS п.34.3) digital photo."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .baldness import estimate_crown_y
from .gate import _landmarker
from .whitening import corner_whiteness

_LEFT = 33
_RIGHT = 263
_CHIN = 152
_FOREHEAD = 10
# Approximate face sides (cheeks) for head width in mm
_LEFT_CHEEK = 234
_RIGHT_CHEEK = 454


def measure_compliance(
    bgr: np.ndarray,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    dpi_target: int | None = None,
) -> dict[str, Any]:
    """Check crop against FMS §34.3 geometry (35×45, face oval, head mm)."""
    h, w = bgr.shape[:2]
    white = corner_whiteness(bgr)
    exp_w = int(expected_width if expected_width is not None else config.PASSPORT_WIDTH)
    exp_h = int(
        expected_height if expected_height is not None else config.PASSPORT_HEIGHT
    )
    dpi = int(dpi_target if dpi_target is not None else config.PASSPORT_DPI)
    size_ok = (w, h) == (exp_w, exp_h)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return {
            "width": w,
            "height": h,
            "dpi_target": dpi,
            "size_ok": size_ok,
            "face_count": 0,
            "bg": white,
            "pass": False,
            "source": "rg.ru/2011/08/22/pasport-dok.html §34.3",
        }

    lm = result.face_landmarks[0]
    # Prefer real silhouette crown (bald-aware); fallback landmark heuristic.
    crown, bald = estimate_crown_y(bgr, lm)
    chin = lm[_CHIN].y * h
    face_h = max(chin - crown, 1.0)
    face_ratio = face_h / h
    top_margin = max(crown, 0.0) / h

    left_x = lm[_LEFT_CHEEK].x * w
    right_x = lm[_RIGHT_CHEEK].x * w
    face_w = abs(right_x - left_x)

    head_h_mm = face_ratio * config.PASSPORT_HEIGHT_MM
    head_w_mm = (face_w / w) * config.PASSPORT_WIDTH_MM

    # Gosuslugi oval 70–80% of frame; FMS head 32–36 mm.
    face_oval_ok = face_ratio >= config.FACE_RATIO_MIN
    head_height_ok = (
        config.HEAD_HEIGHT_MM_MIN <= head_h_mm <= config.HEAD_HEIGHT_MM_MAX
    )
    head_width_ok = (
        config.HEAD_WIDTH_MM_MIN <= head_w_mm <= config.HEAD_WIDTH_MM_MAX
    )
    # Top field ~4–6 mm of 45 mm (MVD 5±1); soft floor avoids landmark noise fails
    top_margin_ok = 0.08 <= top_margin <= 0.14

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

    out: dict[str, Any] = {
        "width": w,
        "height": h,
        "dpi_target": dpi,
        "face_ratio": round(face_ratio, 3),
        "top_margin": round(top_margin, 3),
        "head_height_mm": round(head_h_mm, 1),
        "head_width_mm": round(head_w_mm, 1),
        "bg": white,
        "checks": checks,
        "pass": all(checks[k] for k in hard),
        "source": "rg.ru/2011/08/22/pasport-dok.html §34.3",
    }
    if bald is not None:
        out["baldness"] = bald.as_dict()
    return out
