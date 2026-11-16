"""Detect bangs (чёлка) covering eyebrows — Gosuslugi face-oval fail."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# MediaPipe Face Mesh — eyebrow arcs (subject left / right).
_LEFT_BROW = (70, 63, 105, 66, 107, 55, 65, 52, 53, 46)
_RIGHT_BROW = (300, 293, 334, 296, 336, 285, 295, 282, 283, 276)
_FOREHEAD = 10
_NOSE_TIP = 1

# Dark hair vs skin on a passport selfie (gray luma).
_HAIR_LUMA_MAX = 118
# Fraction of brow-band pixels that look like hair.
DEFAULT_BANGS_FRAC_MIN = 0.42


def _xy(landmarks, idx: int, w: int, h: int) -> tuple[int, int]:
    lm = landmarks[idx]
    return int(round(lm.x * w)), int(round(lm.y * h))


def bangs_on_brows_score(
    bgr: np.ndarray,
    landmarks,
) -> dict[str, Any]:
    """Score how much hair covers the eyebrow band.

    Returns metrics including ``frac`` (0..1 hair in the brow strip) and
    ``bangs_on_brows`` bool using ``DEFAULT_BANGS_FRAC_MIN``.
    """
    h, w = bgr.shape[:2]
    if landmarks is None or len(landmarks) < 337:
        return {"frac": 0.0, "bangs_on_brows": False, "pix": 0, "reason": "no_landmarks"}

    brow_ids = _LEFT_BROW + _RIGHT_BROW
    brow_pts = np.array([_xy(landmarks, i, w, h) for i in brow_ids], dtype=np.int32)
    forehead = _xy(landmarks, _FOREHEAD, w, h)
    nose = _xy(landmarks, _NOSE_TIP, w, h)

    brow_y = float(np.median(brow_pts[:, 1]))
    brow_x0 = int(np.min(brow_pts[:, 0]))
    brow_x1 = int(np.max(brow_pts[:, 0]))
    face_h = max(8.0, float(nose[1] - forehead[1]))

    # Strip from a bit above the brows down onto the brow line (where long fringe lands).
    pad_y = max(2, int(0.04 * face_h))
    y0 = int(max(0, brow_y - 0.12 * face_h))
    y1 = int(min(h, brow_y + pad_y))
    x0 = int(max(0, brow_x0 - 0.02 * w))
    x1 = int(min(w, brow_x1 + 0.02 * w))
    if y1 <= y0 + 2 or x1 <= x0 + 2:
        return {"frac": 0.0, "bangs_on_brows": False, "pix": 0, "reason": "bad_box"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    roi = gray[y0:y1, x0:x1]
    # Exclude near-white background crumbs at the edges of the crop.
    usable = roi < 245
    n = int(usable.sum())
    if n < 40:
        return {"frac": 0.0, "bangs_on_brows": False, "pix": n, "reason": "too_few_pix"}

    hair = usable & (roi < _HAIR_LUMA_MAX)
    frac = float(hair.sum()) / float(n)
    hit = frac >= DEFAULT_BANGS_FRAC_MIN
    return {
        "frac": round(frac, 3),
        "bangs_on_brows": hit,
        "pix": n,
        "thresh": DEFAULT_BANGS_FRAC_MIN,
        "box": [x0, y0, x1, y1],
        "brow_y": round(brow_y, 1),
        "reason": "ok",
    }


def bangs_on_brows(
    bgr: np.ndarray,
    landmarks,
    *,
    frac_min: float | None = None,
) -> bool:
    scores = bangs_on_brows_score(bgr, landmarks)
    if frac_min is None:
        return bool(scores.get("bangs_on_brows"))
    return float(scores.get("frac") or 0.0) >= float(frac_min)
