"""Fast bald / short-hair cue for passport crop (no extra neural net).

Uses MediaPipe forehead/chin already loaded by gate + the white-bg silhouette
after riverflow. Crown gap above forehead → crown_factor hint for crop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker

_LEFT_EYE = 33
_RIGHT_EYE = 263
_CHIN = 152
_FOREHEAD = 10

# (forehead_y - crown_y) / (chin_y - forehead_y)
# Bald / buzz: small gap. Average hair: ~0.40–0.55.
BALD_GAP_RATIO_MAX = 0.30
DEFAULT_CROWN_FACTOR = 0.45
MIN_CROWN_FACTOR = 0.14
MAX_CROWN_FACTOR = 0.65


@dataclass(frozen=True)
class BaldnessAnalysis:
    is_bald: bool
    confidence: float
    gap_ratio: float
    crown_factor: float
    source: str  # silhouette | heuristic | no_face

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def person_mask(bgr: np.ndarray, *, tol: int = 40) -> np.ndarray:
    """True where pixel is not near-white (subject on passport bg)."""
    return (255 - bgr.astype(np.int16)).max(axis=2) > tol


def silhouette_top_y(
    bgr: np.ndarray,
    *,
    mid_x: float,
    half_width: float,
    tol: int = 40,
) -> float | None:
    """Topmost non-white row in a vertical band around face midline."""
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return None
    x0 = max(0, int(mid_x - half_width))
    x1 = min(w, int(mid_x + half_width) + 1)
    if x1 <= x0:
        return None
    mask = person_mask(bgr, tol=tol)
    band = mask[:, x0:x1]
    rows = np.flatnonzero(band.any(axis=1))
    if rows.size == 0:
        return None
    return float(rows[0])


def classify_gap_ratio(gap_ratio: float) -> BaldnessAnalysis:
    """Map measured crown gap (relative to face span) → bald flag + crop factor."""
    g = float(np.clip(gap_ratio, MIN_CROWN_FACTOR, MAX_CROWN_FACTOR))
    is_bald = gap_ratio <= BALD_GAP_RATIO_MAX
    # Confidence peaks when clearly bald or clearly not.
    if is_bald:
        conf = float(np.clip((BALD_GAP_RATIO_MAX - gap_ratio) / BALD_GAP_RATIO_MAX + 0.35, 0.35, 0.95))
    else:
        conf = float(np.clip((gap_ratio - BALD_GAP_RATIO_MAX) / 0.25 + 0.35, 0.35, 0.95))
    return BaldnessAnalysis(
        is_bald=is_bald,
        confidence=round(conf, 3),
        gap_ratio=round(float(gap_ratio), 3),
        crown_factor=round(g, 3),
        source="silhouette",
    )


def analyze_baldness(bgr: np.ndarray) -> BaldnessAnalysis:
    """Detect bald/short crown for white-bg portrait; cheap CPU path."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return BaldnessAnalysis(
            is_bald=False,
            confidence=0.0,
            gap_ratio=DEFAULT_CROWN_FACTOR,
            crown_factor=DEFAULT_CROWN_FACTOR,
            source="no_face",
        )

    lm = result.face_landmarks[0]
    forehead_y = lm[_FOREHEAD].y * h
    chin_y = lm[_CHIN].y * h
    face_span = max(chin_y - forehead_y, 1.0)
    mid_x = ((lm[_LEFT_EYE].x + lm[_RIGHT_EYE].x) / 2.0) * w

    sil_y = silhouette_top_y(
        bgr,
        mid_x=mid_x,
        half_width=max(face_span * 0.35, w * 0.08),
    )
    if sil_y is None or sil_y >= forehead_y:
        return BaldnessAnalysis(
            is_bald=False,
            confidence=0.2,
            gap_ratio=DEFAULT_CROWN_FACTOR,
            crown_factor=DEFAULT_CROWN_FACTOR,
            source="heuristic",
        )

    gap_ratio = (forehead_y - sil_y) / face_span
    analysis = classify_gap_ratio(gap_ratio)
    return analysis


def estimate_crown_y(
    bgr: np.ndarray,
    lm: Any,
    *,
    crown_factor: float = DEFAULT_CROWN_FACTOR,
) -> tuple[float, BaldnessAnalysis | None]:
    """Crown Y for compliance/crop: prefer silhouette, else landmark heuristic."""
    h, w = bgr.shape[:2]
    forehead_y = lm[_FOREHEAD].y * h
    chin_y = lm[_CHIN].y * h
    face_span = max(chin_y - forehead_y, 1.0)
    mid_x = ((lm[_LEFT_EYE].x + lm[_RIGHT_EYE].x) / 2.0) * w
    sil_y = silhouette_top_y(
        bgr,
        mid_x=mid_x,
        half_width=max(face_span * 0.35, w * 0.08),
    )
    if sil_y is not None and sil_y < forehead_y:
        gap_ratio = (forehead_y - sil_y) / face_span
        return float(sil_y), classify_gap_ratio(gap_ratio)
    return forehead_y - float(crown_factor) * face_span, None
