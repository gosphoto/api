"""Detect bangs (чёлка) covering eyebrows — Gosuslugi face-oval fail."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# MediaPipe — eyes anchor the anatomical brow line (brow landmarks drift under fringe).
_LEFT_EYE = (159, 158, 157)
_RIGHT_EYE = (386, 385, 384)
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_FOREHEAD = 10
_NOSE_TIP = 1

# Brow line = eye_y - k * face_height; analyze the strip strictly above it.
_BROW_LINE_K = 0.11
# Fraction of supra-brow strip that looks like hair.
DEFAULT_BANGS_FRAC_MIN = 0.38
DEFAULT_BANGS_CENTER_MIN = 0.45
# Smooth mid-tone "dark" with skin chroma → wet skin / soft shadow, not fringe.
_SMOOTH_EDGE_MAX = 30.0


def _xy(landmarks, idx: int, w: int, h: int) -> tuple[int, int]:
    lm = landmarks[idx]
    return int(round(lm.x * w)), int(round(lm.y * h))


def bangs_on_brows_score(
    bgr: np.ndarray,
    landmarks,
) -> dict[str, Any]:
    """Score whether fringe sits on the eyebrow line.

    Uses an eye-anchored brow line (not MediaPipe brow points — those ride up
    onto the fringe). Rejects short bangs with a skin gap above the brows and
    wet-skin / cap-brim false positives via chroma + edge energy.
    """
    h, w = bgr.shape[:2]
    if landmarks is None or len(landmarks) < 400:
        return {"frac": 0.0, "bangs_on_brows": False, "pix": 0, "reason": "no_landmarks"}

    forehead = _xy(landmarks, _FOREHEAD, w, h)
    nose = _xy(landmarks, _NOSE_TIP, w, h)
    face_h = max(8.0, float(nose[1] - forehead[1]))

    eye_pts = np.array(
        [_xy(landmarks, i, w, h) for i in (_LEFT_EYE + _RIGHT_EYE)],
        dtype=np.int32,
    )
    eye_y = float(np.median(eye_pts[:, 1]))
    brow_line = eye_y - _BROW_LINE_K * face_h

    le = _xy(landmarks, _LEFT_EYE_OUTER, w, h)
    re = _xy(landmarks, _RIGHT_EYE_OUTER, w, h)
    span = max(1, abs(re[0] - le[0]))
    x0 = int(min(le[0], re[0]) + 0.06 * span)
    x1 = int(max(le[0], re[0]) - 0.06 * span)

    y1 = int(max(0, brow_line - 1))
    y0 = int(max(0, brow_line - max(6, 0.14 * face_h)))
    if y1 <= y0 + 2 or x1 <= x0 + 4:
        return {"frac": 0.0, "bangs_on_brows": False, "pix": 0, "reason": "bad_box"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float32)

    cx0, cx1 = int(0.35 * w), int(0.65 * w)
    cy0 = int(nose[1])
    cy1 = int(min(h, nose[1] + 0.18 * face_h))
    cheek = gray[cy0:cy1, cx0:cx1]
    cheek_a = a_ch[cy0:cy1, cx0:cx1]
    skin = float(np.median(cheek)) if cheek.size else 160.0
    skin_a = float(np.median(cheek_a)) if cheek_a.size else 140.0
    thr = max(65.0, min(125.0, skin - 40.0))

    roi = gray[y0:y1, x0:x1]
    roi_a = a_ch[y0:y1, x0:x1]
    usable = roi < 245
    n = int(usable.sum())
    if n < 25:
        return {"frac": 0.0, "bangs_on_brows": False, "pix": n, "reason": "too_few_pix"}

    # Dark hair + mild blonde fringe (slightly darker than cheek skin).
    hair = usable & ((roi < thr) | ((roi < skin - 12) & (roi >= thr)))
    frac = float(hair.sum()) / float(n)

    mid0 = int(0.25 * (x1 - x0))
    mid1 = int(0.75 * (x1 - x0))
    c_frac = float(hair[:, mid0:mid1].sum()) / float(
        max(1, int(usable[:, mid0:mid1].sum()))
    )

    fh_y0 = int(max(0, forehead[1] - 0.02 * face_h))
    fh_y1 = int(min(y0, forehead[1] + 0.1 * face_h))
    fh_x0 = int(x0 + 0.15 * (x1 - x0))
    fh_x1 = int(x1 - 0.15 * (x1 - x0))
    fh_frac = 0.0
    if fh_y1 > fh_y0 + 2 and fh_x1 > fh_x0 + 2:
        fh = gray[fh_y0:fh_y1, fh_x0:fh_x1]
        fu = fh < 245
        fn = int(fu.sum())
        if fn:
            fh_frac = float((fu & ((fh < thr) | (fh < skin - 12))).sum()) / float(fn)

    # Short fringe above brows leaves a skin row in the lower half of the strip.
    rows: list[float] = []
    for yy in range(y0, y1):
        row = gray[yy, x0:x1]
        ru = row < 245
        rn = int(ru.sum())
        if rn < 8:
            continue
        rows.append(float((ru & (row < thr)).sum()) / float(rn))
    lower = rows[len(rows) // 2 :] if rows else []
    skin_gap = bool(lower) and (min(lower) < 0.18)

    sob = cv2.Sobel(roi.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sob = sob + cv2.Sobel(roi.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(np.abs(sob)[usable])) if n else 0.0

    dark = usable & (roi < thr)
    dark_n = int(dark.sum())
    skinish = 0.0
    if dark_n:
        skinish = int((dark & (roi_a >= skin_a - 3)).sum()) / float(dark_n)
    band_med = float(np.median(roi[usable])) if n else 255.0
    smooth_skin = skinish >= 0.5 and edge < _SMOOTH_EDGE_MAX and band_med > 55.0
    brim = band_med < 40.0 and edge < 12.0

    raw = (c_frac >= DEFAULT_BANGS_CENTER_MIN and frac >= DEFAULT_BANGS_FRAC_MIN) and (
        fh_frac >= 0.25 or c_frac >= 0.62
    )
    hit = bool(raw) and not skin_gap and not smooth_skin and not brim

    return {
        "frac": round(frac, 3),
        "c_frac": round(c_frac, 3),
        "fh_frac": round(fh_frac, 3),
        "bangs_on_brows": hit,
        "pix": n,
        "thresh": DEFAULT_BANGS_FRAC_MIN,
        "center_min": DEFAULT_BANGS_CENTER_MIN,
        "box": [x0, y0, x1, y1],
        "brow_line": round(brow_line, 1),
        "eye_y": round(eye_y, 1),
        "edge": round(edge, 1),
        "skin_gap": skin_gap,
        "smooth_skin": smooth_skin,
        "brim": brim,
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
