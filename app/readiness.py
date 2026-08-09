"""Cheap studio-readiness check: skip Riverflow when photo is already clean white-bg.

CPU-only (OpenCV). Fail-closed: any doubt → full edit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np

# Border / bg thresholds (fail-closed).
CORNER_WHITE_MIN = 245.0
BORDER_WHITE_MIN = 242.0
BORDER_WHITE_FRAC_MIN = 0.92
BG_LUMA_STD_MAX = 12.0
BG_CHROMA_MEAN_MAX = 8.0
SUBJECT_AREA_FRAC_MIN = 0.08
SUBJECT_AREA_FRAC_MAX = 0.85


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    reason: str
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    source: str = "cpu_studio"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _person_mask_near_white(bgr: np.ndarray, *, tol: int = 40) -> np.ndarray:
    """Subject ≈ not near-white (works after studio bg or pure white canvas)."""
    return (255 - bgr.astype(np.int16)).max(axis=2) > tol


def _corner_whiteness(bgr: np.ndarray) -> dict[str, Any]:
    """Corner mean BGR; bottom corners ignore dark clothing."""
    n = 12
    chips = [
        bgr[:n, :n],
        bgr[:n, -n:],
        bgr[-n:, :n],
        bgr[-n:, -n:],
    ]
    means: list[np.ndarray] = []
    for i, c in enumerate(chips):
        pix = c.reshape(-1, 3).astype(np.float32)
        luma = pix.mean(axis=1)
        if i >= 2:
            keep = luma >= 210
            if int(keep.sum()) < max(4, pix.shape[0] // 4):
                continue
            pix = pix[keep]
        means.append(pix.mean(axis=0))
    if not means:
        means = [c.reshape(-1, 3).mean(axis=0) for c in chips[:2]]
    avg = np.mean(means, axis=0)
    return {
        "bgr": [round(float(x), 1) for x in avg],
        "white_ok": bool(np.all(avg >= CORNER_WHITE_MIN)),
    }


def _border_whiteness(bgr: np.ndarray, *, band_frac: float = 0.06) -> dict[str, Any]:
    h, w = bgr.shape[:2]
    band = max(6, int(band_frac * min(h, w)))
    strips = [
        bgr[:band, :],
        bgr[-band:, :],
        bgr[:, :band],
        bgr[:, -band:],
    ]
    pix = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0).astype(np.float32)
    luma = pix.mean(axis=1)
    chroma = np.linalg.norm(pix - pix.mean(axis=1, keepdims=True), axis=1)
    white = (luma >= BORDER_WHITE_MIN) & (chroma <= 14.0)
    frac = float(white.mean()) if pix.size else 0.0
    return {
        "border_white_frac": round(frac, 3),
        "border_luma_mean": round(float(luma.mean()), 1),
        "border_ok": frac >= BORDER_WHITE_FRAC_MIN,
    }


def _bg_uniformity(bgr: np.ndarray, subject: np.ndarray) -> dict[str, Any]:
    bg = ~subject
    if int(bg.sum()) < 50:
        return {
            "bg_luma_std": 999.0,
            "bg_chroma_mean": 999.0,
            "bg_uniform_ok": False,
        }
    f = bgr.astype(np.float32)
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_c = f.mean(axis=2)
    chroma = np.linalg.norm(f - mean_c[:, :, None], axis=2)
    bg_luma = luma[bg]
    bg_chroma = chroma[bg]
    luma_std = float(bg_luma.std())
    chroma_mean = float(bg_chroma.mean())
    return {
        "bg_luma_std": round(luma_std, 2),
        "bg_chroma_mean": round(chroma_mean, 2),
        "bg_uniform_ok": luma_std <= BG_LUMA_STD_MAX and chroma_mean <= BG_CHROMA_MEAN_MAX,
    }


def assess_readiness(bgr: np.ndarray) -> ReadinessResult:
    """Return ready=True only when crop-only path is safe enough."""
    if bgr is None or bgr.size == 0 or bgr.ndim != 3:
        return ReadinessResult(
            ready=False,
            reason="empty",
            reasons=["empty"],
            scores={},
        )

    h, w = bgr.shape[:2]
    subject = _person_mask_near_white(bgr)
    area = float(subject.mean()) if subject.size else 0.0
    corners = _corner_whiteness(bgr)
    border = _border_whiteness(bgr)
    uniform = _bg_uniformity(bgr, subject)

    scores: dict[str, Any] = {
        "corner_white_ok": bool(corners.get("white_ok")),
        "corner_bgr": corners.get("bgr"),
        "subject_area_frac": round(area, 3),
        **border,
        **uniform,
    }

    reasons: list[str] = []
    if not scores["corner_white_ok"]:
        reasons.append("bg_not_white")
    if not border["border_ok"]:
        reasons.append("border_not_white")
    if not uniform["bg_uniform_ok"]:
        reasons.append("bg_not_uniform")
    if area < SUBJECT_AREA_FRAC_MIN:
        reasons.append("subject_too_small")
    if area > SUBJECT_AREA_FRAC_MAX:
        reasons.append("subject_too_large")

    if reasons:
        return ReadinessResult(
            ready=False,
            reason=reasons[0],
            reasons=reasons,
            scores=scores,
        )
    return ReadinessResult(
        ready=True,
        reason="studio_ready",
        reasons=[],
        scores=scores,
    )
