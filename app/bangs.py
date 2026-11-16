"""Detect bangs (чёлка) covering eyebrows — Gosuslugi face-oval fail."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Eyes anchor the anatomical brow line (brow landmarks drift under fringe).
_LEFT_EYE = (159, 158, 157)
_RIGHT_EYE = (386, 385, 384)
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_OUTER = 263
_FOREHEAD = 10
_NOSE_TIP = 1

_BROW_LINE_K = 0.11
# True fringe must cover mid-forehead, not only cast a brow-line shadow.
DEFAULT_MID_FRAC_MIN = 0.65
DEFAULT_BANGS_FRAC_MIN = 0.40
DEFAULT_BANGS_CENTER_MIN = 0.48


def _xy(landmarks, idx: int, w: int, h: int) -> tuple[int, int]:
    lm = landmarks[idx]
    return int(round(lm.x * w)), int(round(lm.y * h))


def bangs_on_brows_score(
    bgr: np.ndarray,
    landmarks,
) -> dict[str, Any]:
    """Score whether fringe covers the forehead down onto the brows.

    Requires dark hair in the mid-forehead plane (чёлка), not merely a dark
    strip at the brow line (eyebrows, short fringe with skin gap, wet skin).
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
    # Upside-down / extreme pitch: nose should sit below the eyes.
    if float(nose[1]) < eye_y - 0.02 * face_h:
        return {
            "frac": 0.0,
            "bangs_on_brows": False,
            "pix": 0,
            "reason": "upside_down",
        }
    brow_line = eye_y - _BROW_LINE_K * face_h

    le = _xy(landmarks, _LEFT_EYE_OUTER, w, h)
    re = _xy(landmarks, _RIGHT_EYE_OUTER, w, h)
    span = max(1, abs(re[0] - le[0]))
    x0 = int(min(le[0], re[0]) + 0.08 * span)
    x1 = int(max(le[0], re[0]) - 0.08 * span)

    y1 = int(max(0, brow_line - 1))
    y0 = int(max(0, brow_line - max(6, 0.12 * face_h)))
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

    def _hair_frac(ya: int, yb: int, xa: int, xb: int) -> tuple[float, float, int]:
        if yb <= ya + 1 or xb <= xa + 2:
            return 0.0, 0.0, 0
        roi = gray[ya:yb, xa:xb]
        roi_a = a_ch[ya:yb, xa:xb]
        usable = roi < 245
        n = int(usable.sum())
        if n < 20:
            return 0.0, 0.0, n
        hair = usable & ((roi < thr) | ((roi < skin - 12) & (roi >= thr)))
        dark = usable & (roi < thr)
        skinish = 0.0
        dn = int(dark.sum())
        if dn:
            skinish = int((dark & (roi_a >= skin_a - 3)).sum()) / float(dn)
        return float(hair.sum()) / float(n), skinish, n

    frac, skinish, n = _hair_frac(y0, y1, x0, x1)
    mid0 = int(0.25 * (x1 - x0))
    mid1 = int(0.75 * (x1 - x0))
    c_frac, _, _ = _hair_frac(y0, y1, x0 + mid0, x0 + mid1)

    fh_top = float(forehead[1])
    gap = max(4.0, brow_line - fh_top)
    mid_y0 = int(fh_top + 0.28 * gap)
    mid_y1 = int(fh_top + 0.72 * gap)
    # Stricter darkness on mid-forehead (чёлка), vs soft shadows on bare skin.
    thr_mid = max(55.0, min(115.0, skin - 50.0))
    mid_roi = gray[mid_y0:mid_y1, x0:x1]
    mid_u = mid_roi < 245
    mid_n = int(mid_u.sum())
    if mid_n < 20:
        mid_frac = 0.0
        mid_skinish = 0.0
        mid_p25 = skin
        mid_med = skin
    else:
        mid_hair = mid_u & ((mid_roi < thr_mid) | ((mid_roi < skin - 18) & (mid_roi >= thr_mid)))
        mid_frac = float(mid_hair.sum()) / float(mid_n)
        mid_dark = mid_u & (mid_roi < thr_mid)
        dn = int(mid_dark.sum())
        mid_a = a_ch[mid_y0:mid_y1, x0:x1]
        mid_skinish = (
            int((mid_dark & (mid_a >= skin_a - 3)).sum()) / float(dn) if dn else 0.0
        )
        mid_p25 = float(np.percentile(mid_roi[mid_u], 25))
        mid_med = float(np.median(mid_roi[mid_u]))

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

    sob_roi = gray[y0:y1, x0:x1]
    usable = sob_roi < 245
    sob = cv2.Sobel(sob_roi.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    sob = sob + cv2.Sobel(sob_roi.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    edge = float(np.mean(np.abs(sob)[usable])) if int(usable.sum()) else 0.0
    band_med = float(np.median(sob_roi[usable])) if int(usable.sum()) else 255.0
    brim = band_med < 40.0 and edge < 12.0
    wet_mid = mid_skinish >= 0.70 and edge < 25.0 and band_med > 70.0
    # Bare forehead with only mild underexposure is not fringe.
    weak_mid = (skin - mid_p25) < 48.0

    hit = (
        mid_frac >= DEFAULT_MID_FRAC_MIN
        and c_frac >= DEFAULT_BANGS_CENTER_MIN
        and frac >= DEFAULT_BANGS_FRAC_MIN
        and not skin_gap
        and not brim
        and not wet_mid
        and not weak_mid
    )

    return {
        "frac": round(frac, 3),
        "c_frac": round(c_frac, 3),
        "mid_frac": round(mid_frac, 3),
        "mid_skinish": round(mid_skinish, 3),
        "mid_p25": round(mid_p25, 1),
        "weak_mid": weak_mid,
        "bangs_on_brows": hit,
        "pix": n,
        "thresh": DEFAULT_BANGS_FRAC_MIN,
        "mid_min": DEFAULT_MID_FRAC_MIN,
        "center_min": DEFAULT_BANGS_CENTER_MIN,
        "box": [x0, y0, x1, y1],
        "mid_box": [x0, mid_y0, x1, mid_y1],
        "brow_line": round(brow_line, 1),
        "eye_y": round(eye_y, 1),
        "edge": round(edge, 1),
        "skin_gap": skin_gap,
        "wet_mid": wet_mid,
        "brim": brim,
        "skinish": round(skinish, 3),
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
