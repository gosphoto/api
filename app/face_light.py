"""CPU flatten for leftover one-sided face light after generative edit.

Gemini often keeps the source selfie’s left/right cheek gradient. Gosuslugi
rejects that as «неравномерно освещено». Brighten the darker face half inside
the MediaPipe face oval until both halves match — no second model call.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .face_protect import _landmarks_xy, face_protect_mask

# MediaPipe nose tip.
_NOSE = 1

# Apply only when cheek L-gap is clearly one-sided.
_DELTA_MIN = 12.0
_MAX_GAIN = 50.0
_MAX_ITERS = 3


def _lr_means(
    L: np.ndarray, mask: np.ndarray, nose_x: float
) -> tuple[float | None, float | None]:
    _h, w = L.shape
    xs = np.arange(w, dtype=np.float32)[None, :]
    left = mask & (xs < nose_x) & (L < 245) & (L > 35)
    right = mask & (xs >= nose_x) & (L < 245) & (L > 35)

    def mean(sel: np.ndarray) -> float | None:
        if int(sel.sum()) < 40:
            return None
        return float(L[sel].mean())

    return mean(left), mean(right)


def measure_face_lr_delta(bgr: np.ndarray) -> dict[str, Any] | None:
    """Left/right face L means inside the oval. None if no face."""
    lm = _landmarks_xy(bgr)
    if lm is None:
        return None
    mask_f = face_protect_mask(bgr)
    if mask_f is None:
        return None
    hard = mask_f > 0.35
    nose_x = float(lm[_NOSE][0]) if lm.shape[0] > _NOSE else float(lm[:, 0].mean())
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ml, mr = _lr_means(lab[:, :, 0], hard, nose_x)
    if ml is None or mr is None:
        return None
    return {
        "L_left": round(ml, 1),
        "L_right": round(mr, 1),
        "delta": round(ml - mr, 1),
        "abs_delta": round(abs(ml - mr), 1),
    }


def flatten_face_lighting(
    bgr: np.ndarray,
    *,
    delta_min: float = _DELTA_MIN,
    max_gain: float = _MAX_GAIN,
    max_iters: int = _MAX_ITERS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Brighten the darker face half to match the brighter one.

    No-op when landmarks are missing or abs(delta) < delta_min.
    """
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return bgr, {"applied": False, "reason": "empty"}

    lm = _landmarks_xy(bgr)
    if lm is None:
        return bgr, {"applied": False, "reason": "no_face"}

    mask_f = face_protect_mask(bgr)
    if mask_f is None:
        return bgr, {"applied": False, "reason": "no_mask"}

    ys = lm[:, 1]
    xs = lm[:, 0]
    chin = float(ys.max())
    cx = float(xs.mean())
    fw = float(max(xs.max() - xs.min(), 1.0))
    h, w = bgr.shape[:2]

    # Soft under-chin band — side-key often leaves a dark neck half.
    neck = np.zeros((h, w), np.float32)
    y0 = int(chin)
    y1 = min(h, int(chin + fw * 0.22))
    x0 = max(0, int(cx - fw * 0.38))
    x1 = min(w, int(cx + fw * 0.38))
    if y1 > y0 and x1 > x0:
        neck[y0:y1, x0:x1] = 0.55
        neck = cv2.GaussianBlur(neck, (0, 0), max(3.0, fw * 0.04))

    apply_m = np.clip(np.maximum(mask_f, neck), 0.0, 1.0)
    hard = apply_m > 0.35
    nose_x = float(lm[_NOSE][0]) if lm.shape[0] > _NOSE else cx

    out = bgr.copy()
    meta: dict[str, Any] = {"applied": False, "iters": []}
    before = _lr_means(
        cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0],
        hard,
        nose_x,
    )
    if before[0] is not None and before[1] is not None:
        meta["before"] = {
            "L_left": round(before[0], 1),
            "L_right": round(before[1], 1),
            "delta": round(before[0] - before[1], 1),
        }
        if abs(before[0] - before[1]) < delta_min:
            meta["reason"] = "already_even"
            return out, meta

    for _ in range(max(1, int(max_iters))):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        ml, mr = _lr_means(L, hard, nose_x)
        if ml is None or mr is None:
            meta["reason"] = "empty_half"
            break
        delta = ml - mr
        meta["iters"].append(
            {"L_left": round(ml, 1), "L_right": round(mr, 1), "delta": round(delta, 1)}
        )
        if abs(delta) < delta_min:
            break

        target = max(ml, mr)
        g_left = min(max_gain, max(0.0, target - ml))
        g_right = min(max_gain, max(0.0, target - mr))
        xx = np.arange(w, dtype=np.float32)
        sigma = max(8.0, fw * 0.08)
        t = 1.0 / (1.0 + np.exp(-(xx - nose_x) / sigma))
        gain_row = g_left * (1.0 - t) + g_right * t
        gain = np.tile(gain_row, (h, 1)) * apply_m

        # Local fill for residual dark pockets (under-eye / nose cast).
        dark = hard & (L < (target - 8.0))
        extra = np.zeros_like(L)
        if dark.any():
            extra[dark] = np.clip((target - 4.0) - L[dark], 0.0, max_gain * 0.55)
            extra = cv2.GaussianBlur(extra, (0, 0), max(2.0, fw * 0.03))

        lab[:, :, 0] = np.clip(L + gain + extra * apply_m, 0, 255)
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        meta["applied"] = True

    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    ml, mr = _lr_means(lab[:, :, 0], hard, nose_x)
    if ml is not None and mr is not None:
        meta["after"] = {
            "L_left": round(ml, 1),
            "L_right": round(mr, 1),
            "delta": round(ml - mr, 1),
        }
    return out, meta
