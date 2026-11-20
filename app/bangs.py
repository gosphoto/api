"""Detect bangs covering brows/eyes via BiSeNet fringe ∩ face oval."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import config
from . import face_parse as fp

# Calibrated on bangs_gate overlays (hair|hat as fringe).
# Light brow tip-touch must PASS; only clear cover → reject.
DEFAULT_MID_MIN = 0.30
DEFAULT_FALLBACK_MIN = 0.35
DEFAULT_SKIN_TOUCH_MIN = 0.28
DEFAULT_BROW_TOUCH_MIN = 0.33
DEFAULT_BROWS_MISSING_MID_MIN = 0.25
DEFAULT_BROW_DILATE_PX = 10
DEFAULT_SKIN_DILATE_PX = 6


def _dilate(binary: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return binary.astype(bool)
    k = 2 * int(radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool)


def _part_mask(mask: np.ndarray, ids: int | tuple[int, ...]) -> np.ndarray:
    if isinstance(ids, tuple):
        out = np.zeros(mask.shape[:2], dtype=bool)
        for i in ids:
            out |= mask == i
        return out
    return mask == ids


def hair_oval_intersection(
    mask: np.ndarray,
    *,
    mid_min: float | None = None,
    fallback_min: float | None = None,
    skin_touch_min: float | None = None,
    brow_touch_min: float | None = None,
) -> dict[str, Any]:
    """Score fringe (hair|hat) covering brows/eyes on a BiSeNet class mask.

    Classes are mutually exclusive, so we measure:
    - mid: fringe in a band just above the eyes (center)
    - brow_touch: fringe inside dilated brow neighborhood
    - skin_touch: fringe along upper skin (forehead) boundary
    - fallback: upper-center fringe when eyes are barely parsed
    - brows_missing: eyes present but brows nearly gone (covered)
    """
    if mask is None or mask.size == 0:
        return {
            "hit": False,
            "bangs_on_brows": False,
            "ratio": 0.0,
            "frac": 0.0,
            "reason": "empty_mask",
        }

    t_mid = float(
        mid_min
        if mid_min is not None
        else getattr(config, "BANGS_MID_MIN", DEFAULT_MID_MIN)
    )
    t_fb = float(
        fallback_min
        if fallback_min is not None
        else getattr(config, "BANGS_FALLBACK_MIN", DEFAULT_FALLBACK_MIN)
    )
    t_st = float(
        skin_touch_min
        if skin_touch_min is not None
        else getattr(config, "BANGS_SKIN_TOUCH_MIN", DEFAULT_SKIN_TOUCH_MIN)
    )
    t_bt = float(
        brow_touch_min
        if brow_touch_min is not None
        else getattr(config, "BANGS_BROW_TOUCH_MIN", DEFAULT_BROW_TOUCH_MIN)
    )
    t_miss_mid = float(
        getattr(config, "BANGS_BROWS_MISSING_MID_MIN", DEFAULT_BROWS_MISSING_MID_MIN)
    )

    fringe = np.isin(mask, [fp.CLASS_HAIR, 18])  # hair + hat (fringe mislabel)
    brows = np.isin(mask, [fp.CLASS_L_BROW, fp.CLASS_R_BROW])
    eyes = np.isin(mask, [fp.CLASS_L_EYE, fp.CLASS_R_EYE])
    skin = mask == fp.CLASS_SKIN
    eye_n = int(eyes.sum())
    brow_n = int(brows.sum())
    h, w = mask.shape[:2]

    brow_r = int(getattr(config, "BANGS_BROW_DILATE_PX", DEFAULT_BROW_DILATE_PX))
    if brow_n:
        brows_d = _dilate(brows, brow_r)
        brow_touch = float((fringe & brows_d).sum()) / float(max(int(brows_d.sum()), 1))
    else:
        brow_touch = 0.0

    mid = 0.0
    if eye_n > 20:
        ey = int(np.percentile(np.where(eyes)[0], 15))
        xs = np.where(eyes)[1]
        x0, x1 = int(xs.min()), int(xs.max())
        fw = max(1, x1 - x0)
        band = mask[
            max(0, ey - int(0.10 * h)) : ey,
            max(0, x0 - int(0.15 * fw)) : min(w, x1 + int(0.15 * fw)),
        ]
        if band.size:
            mid = float(np.isin(band, [fp.CLASS_HAIR, 18]).mean())

    brows_missing = eye_n > 50 and brow_n < 0.2 * eye_n

    fallback = 0.0
    if eye_n < 40:
        ys, xs = np.where(skin | fringe)
        if len(ys) > 50:
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            fh = max(1, y1 - y0)
            fw = max(1, x1 - x0)
            band = mask[
                y0 : y0 + int(0.45 * fh),
                x0 + int(0.2 * fw) : x0 + int(0.8 * fw),
            ]
            if band.size:
                fallback = float(np.isin(band, [fp.CLASS_HAIR, 18]).mean())

    skin_touch = 0.0
    if int(skin.sum()) > 100:
        ys, xs = np.where(skin)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        fh = max(1, y1 - y0)
        fw = max(1, x1 - x0)
        region = np.zeros_like(mask, dtype=bool)
        region[
            y0 : y0 + int(0.40 * fh),
            x0 + int(0.2 * fw) : x0 + int(0.8 * fw),
        ] = True
        skin_r = int(getattr(config, "BANGS_SKIN_DILATE_PX", DEFAULT_SKIN_DILATE_PX))
        skin_d = _dilate(skin, skin_r) & region
        skin_touch = float((fringe & skin_d).sum()) / float(max(int(skin_d.sum()), 1))

    hit = (
        mid >= t_mid
        or fallback >= t_fb
        or (brows_missing and mid >= t_miss_mid)
        or skin_touch >= t_st
        or brow_touch >= t_bt
    )

    # Per-part diagnostics (dilated neighborhood fringe fraction)
    parts: dict[str, Any] = {}
    part_ids = {
        "l_brow": fp.CLASS_L_BROW,
        "r_brow": fp.CLASS_R_BROW,
        "l_eye": fp.CLASS_L_EYE,
        "r_eye": fp.CLASS_R_EYE,
        "nose": fp.CLASS_NOSE,
        "mouth": (fp.CLASS_MOUTH, fp.CLASS_U_LIP, fp.CLASS_L_LIP),
    }
    for name, ids in part_ids.items():
        part = _part_mask(mask, ids)
        if int(part.sum()) < 8:
            parts[name] = {
                "ratio": 0.0,
                "intersect_pix": 0,
                "oval_pix": 0,
                "hit": False,
            }
            continue
        part_d = _dilate(part, brow_r)
        inter = int((fringe & part_d).sum())
        opix = int(part_d.sum())
        ratio = float(inter) / float(max(opix, 1))
        parts[name] = {
            "ratio": round(ratio, 4),
            "intersect_pix": inter,
            "oval_pix": opix,
            "hit": bool(ratio >= t_bt and inter >= 8),
        }

    ratio = max(mid, brow_touch, skin_touch, fallback)
    return {
        "hit": bool(hit),
        "bangs_on_brows": bool(hit),
        "ratio": round(ratio, 4),
        "frac": round(ratio, 4),
        "mid": round(mid, 4),
        "brow_touch": round(brow_touch, 4),
        "skin_touch": round(skin_touch, 4),
        "fallback": round(fallback, 4),
        "brows_missing": bool(brows_missing),
        "eye_pix": eye_n,
        "brow_pix": brow_n,
        "hair_pix": int(fringe.sum()),
        "thresh_mid": t_mid,
        "thresh_fallback": t_fb,
        "thresh_skin_touch": t_st,
        "thresh_brow_touch": t_bt,
        "thresh_brows_missing_mid": t_miss_mid,
        "parts": parts,
        "l_brow": parts["l_brow"],
        "r_brow": parts["r_brow"],
        "l_eye": parts["l_eye"],
        "r_eye": parts["r_eye"],
        "nose": parts["nose"],
        "mouth": parts["mouth"],
        "reason": "ok",
    }


def bangs_on_brows_score(
    bgr: np.ndarray,
    landmarks=None,
) -> dict[str, Any]:
    """Run BiSeNet + hair∩oval. ``landmarks`` kept for call-site compat (unused)."""
    del landmarks
    try:
        artifacts = fp.parse_with_artifacts(bgr)
    except FileNotFoundError as e:
        return {
            "frac": 0.0,
            "bangs_on_brows": False,
            "hit": False,
            "pix": 0,
            "reason": "no_model",
            "error": str(e),
        }
    except Exception as e:
        return {
            "frac": 0.0,
            "bangs_on_brows": False,
            "hit": False,
            "pix": 0,
            "reason": "parse_error",
            "error": str(e),
        }

    scores = hair_oval_intersection(artifacts["mask"])
    scores["pix"] = int(scores.get("hair_pix") or 0)
    scores["bisenet_overlay_png"] = artifacts["overlay_png"]
    scores["bisenet_mask_png"] = artifacts["mask_png"]
    return scores


def bangs_on_brows(
    bgr: np.ndarray,
    landmarks=None,
    *,
    frac_min: float | None = None,
) -> bool:
    scores = bangs_on_brows_score(bgr, landmarks)
    if frac_min is None:
        return bool(scores.get("bangs_on_brows"))
    return float(scores.get("ratio") or scores.get("frac") or 0.0) >= float(frac_min)
