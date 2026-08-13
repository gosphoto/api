"""MediaPipe Pose — torso upsell + shoulder roll for passport crop."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from . import config

log = logging.getLogger("gosphoto-gate")

# BlazePose 33-landmark indices
_NOSE = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12


@dataclass
class TorsoResult:
    ok: bool
    reason: str
    metrics: dict[str, Any]


@dataclass
class ShoulderRoll:
    """Roll angle to level the shoulder line (degrees, OpenCV warpAffine sense)."""

    deg: float
    metrics: dict[str, Any]


def normalize_roll_deg(deg: float) -> float:
    """Map roll to (-90, 90] so reversed L/R axes stay near 0, not ±180."""
    while deg > 180.0:
        deg -= 360.0
    while deg <= -180.0:
        deg += 360.0
    if deg > 90.0:
        deg -= 180.0
    elif deg <= -90.0:
        deg += 180.0
    return deg


def shoulder_roll_from_landmarks(
    *,
    left_shoulder_x: float,
    left_shoulder_y: float,
    left_vis: float,
    right_shoulder_x: float,
    right_shoulder_y: float,
    right_vis: float,
    min_visibility: float | None = None,
    min_shoulder_width: float | None = None,
) -> ShoulderRoll | None:
    """Pure: return roll deg to make L→R shoulder line horizontal, or None."""
    min_vis = (
        config.TORSO_MIN_VISIBILITY if min_visibility is None else min_visibility
    )
    min_width = (
        config.TORSO_MIN_SHOULDER_WIDTH
        if min_shoulder_width is None
        else min_shoulder_width
    )
    metrics: dict[str, Any] = {
        "left_shoulder": {
            "x": round(left_shoulder_x, 4),
            "y": round(left_shoulder_y, 4),
            "visibility": round(left_vis, 4),
        },
        "right_shoulder": {
            "x": round(right_shoulder_x, 4),
            "y": round(right_shoulder_y, 4),
            "visibility": round(right_vis, 4),
        },
        "min_visibility": min_vis,
        "min_shoulder_width": min_width,
    }
    if left_vis < min_vis or right_vis < min_vis:
        metrics["reason"] = "shoulders_low_visibility"
        return None
    width = abs(left_shoulder_x - right_shoulder_x)
    metrics["shoulder_width"] = round(width, 4)
    if width < min_width:
        metrics["reason"] = "shoulders_too_narrow"
        return None
    dx = right_shoulder_x - left_shoulder_x
    dy = right_shoulder_y - left_shoulder_y
    # Same convention as eye roll in crop: atan2(dy, dx) → warpAffine angle.
    # Pose L/R are anatomical: facing camera often has dx < 0 → raw ≈ ±180.
    raw = math.degrees(math.atan2(dy, dx))
    deg = normalize_roll_deg(raw)
    metrics["shoulder_roll_raw_deg"] = round(raw, 2)
    metrics["shoulder_roll_deg"] = round(deg, 2)
    metrics["reason"] = "ok"
    return ShoulderRoll(deg=deg, metrics=metrics)


def decide_torso_ok(
    *,
    nose_y: float,
    left_shoulder_x: float,
    left_shoulder_y: float,
    left_vis: float,
    right_shoulder_x: float,
    right_shoulder_y: float,
    right_vis: float,
    min_visibility: float | None = None,
    min_shoulder_drop: float | None = None,
    min_shoulder_width: float | None = None,
) -> TorsoResult:
    """Pure decision from normalized Pose landmarks (y down, x right, 0–1)."""
    min_vis = (
        config.TORSO_MIN_VISIBILITY if min_visibility is None else min_visibility
    )
    min_drop = (
        config.TORSO_MIN_SHOULDER_DROP
        if min_shoulder_drop is None
        else min_shoulder_drop
    )
    min_width = (
        config.TORSO_MIN_SHOULDER_WIDTH
        if min_shoulder_width is None
        else min_shoulder_width
    )

    metrics: dict[str, Any] = {
        "nose_y": round(nose_y, 4),
        "left_shoulder": {
            "x": round(left_shoulder_x, 4),
            "y": round(left_shoulder_y, 4),
            "visibility": round(left_vis, 4),
        },
        "right_shoulder": {
            "x": round(right_shoulder_x, 4),
            "y": round(right_shoulder_y, 4),
            "visibility": round(right_vis, 4),
        },
        "min_visibility": min_vis,
        "min_shoulder_drop": min_drop,
        "min_shoulder_width": min_width,
    }

    if left_vis < min_vis or right_vis < min_vis:
        return TorsoResult(
            ok=False,
            reason="shoulders_low_visibility",
            metrics=metrics,
        )

    mid_y = 0.5 * (left_shoulder_y + right_shoulder_y)
    drop = mid_y - nose_y
    width = abs(left_shoulder_x - right_shoulder_x)
    metrics["shoulder_mid_y"] = round(mid_y, 4)
    metrics["shoulder_drop"] = round(drop, 4)
    metrics["shoulder_width"] = round(width, 4)

    if drop < min_drop:
        return TorsoResult(
            ok=False, reason="shoulders_too_close_to_face", metrics=metrics
        )
    if width < min_width:
        return TorsoResult(ok=False, reason="shoulders_too_narrow", metrics=metrics)

    return TorsoResult(ok=True, reason="torso_ok", metrics=metrics)


@lru_cache(maxsize=1)
def _pose_landmarker():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not config.POSE_MODEL_PATH.is_file():
        log.warning("Pose model not found: %s", config.POSE_MODEL_PATH)
        return None
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(config.POSE_MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def warmup() -> None:
    """Load Pose model once at process start (no-op if missing)."""
    _pose_landmarker()


def _pose_shoulder_landmarks(
    bgr: Any,
) -> tuple[Any, Any] | None:
    """Return (left_shoulder, right_shoulder) Pose landmarks or None."""
    import numpy as np

    landmarker = _pose_landmarker()
    if landmarker is None:
        return None
    import mediapipe as mp

    if bgr is None or getattr(bgr, "size", 0) == 0:
        return None
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect(mp_image)
    except Exception as e:
        log.warning("Pose detect failed: %s", e)
        return None
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    if len(lm) <= _RIGHT_SHOULDER:
        return None
    return lm[_LEFT_SHOULDER], lm[_RIGHT_SHOULDER]


def measure_shoulder_roll(bgr: Any) -> ShoulderRoll | None:
    """
    Detect Pose shoulders and return roll to level them.

    Independent of RESUME_UPSELL_ENABLED — used by passport crop.
    """
    pair = _pose_shoulder_landmarks(bgr)
    if pair is None:
        return None
    ls, rs = pair
    return shoulder_roll_from_landmarks(
        left_shoulder_x=float(ls.x),
        left_shoulder_y=float(ls.y),
        left_vis=float(getattr(ls, "visibility", 0.0) or 0.0),
        right_shoulder_x=float(rs.x),
        right_shoulder_y=float(rs.y),
        right_vis=float(getattr(rs, "visibility", 0.0) or 0.0),
    )


def shoulder_roll_deg(bgr: Any) -> float | None:
    """Degrees to level shoulder line, or None if Pose shoulders unusable."""
    measured = measure_shoulder_roll(bgr)
    return None if measured is None else float(measured.deg)


def assess_torso(data: bytes) -> TorsoResult:
    """Return whether the selfie shows usable upper torso for suit generation."""
    if not config.RESUME_UPSELL_ENABLED:
        return TorsoResult(ok=False, reason="upsell_disabled", metrics={})

    landmarker = _pose_landmarker()
    if landmarker is None:
        return TorsoResult(ok=False, reason="pose_model_missing", metrics={})

    import mediapipe as mp

    from .gate import _decode_image

    bgr = _decode_image(data)
    if bgr is None:
        return TorsoResult(ok=False, reason="decode_error", metrics={})

    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    try:
        result = landmarker.detect(mp_image)
    except Exception as e:
        log.warning("Pose detect failed: %s", e)
        return TorsoResult(
            ok=False, reason="pose_detect_error", metrics={"error": str(e)}
        )

    if not result.pose_landmarks:
        return TorsoResult(ok=False, reason="no_pose", metrics={})

    lm = result.pose_landmarks[0]
    if len(lm) <= _RIGHT_SHOULDER:
        return TorsoResult(ok=False, reason="incomplete_landmarks", metrics={})

    nose = lm[_NOSE]
    ls = lm[_LEFT_SHOULDER]
    rs = lm[_RIGHT_SHOULDER]
    return decide_torso_ok(
        nose_y=float(nose.y),
        left_shoulder_x=float(ls.x),
        left_shoulder_y=float(ls.y),
        left_vis=float(getattr(ls, "visibility", 0.0) or 0.0),
        right_shoulder_x=float(rs.x),
        right_shoulder_y=float(rs.y),
        right_vis=float(getattr(rs, "visibility", 0.0) or 0.0),
    )
