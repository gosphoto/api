"""Paste original face texture onto an edited portrait (anti-beautify)."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker

_LEFT_EYE = 33
_RIGHT_EYE = 263
_CHIN = 152
_NOSE = 1


def _landmarks_xy(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def _face_mask(shape: tuple[int, int], pts: np.ndarray) -> np.ndarray:
    """Soft oval covering full face interior — leave outer hair for edited silhouette."""
    h, w = shape
    xs, ys = pts[:, 0], pts[:, 1]
    face_h = float(max(ys.max() - ys.min(), 1.0))
    face_w = float(max(xs.max() - xs.min(), 1.0))
    cx = float(xs.mean())
    cy = float((ys.min() + ys.max()) * 0.50)
    mask = np.zeros((h, w), np.float32)
    # Cover cheeks/forehead/chin; stay inside hairline so OR fringe cleanup remains
    axes = (max(int(face_w * 0.58), 10), max(int(face_h * 0.72), 10))
    cv2.ellipse(
        mask,
        (int(cx), int(cy)),
        axes,
        0,
        0,
        360,
        1.0,
        -1,
    )
    k = max(21, (int(0.05 * min(h, w)) | 1))
    mask = cv2.GaussianBlur(mask, (k, k), 0)
    return np.clip(mask, 0.0, 1.0)


def restore_face_from_original(
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """
    Warp original face onto edited image so skin texture stays natural.

    Background / silhouette from `edited_bgr`; face interior from `original_bgr`.
    Returns (image, applied).
    """
    src_lm = _landmarks_xy(original_bgr)
    dst_lm = _landmarks_xy(edited_bgr)
    if src_lm is None or dst_lm is None:
        return edited_bgr, False

    src_pts = np.float32(
        [
            src_lm[_LEFT_EYE],
            src_lm[_RIGHT_EYE],
            src_lm[_CHIN],
            src_lm[_NOSE],
        ]
    )
    dst_pts = np.float32(
        [
            dst_lm[_LEFT_EYE],
            dst_lm[_RIGHT_EYE],
            dst_lm[_CHIN],
            dst_lm[_NOSE],
        ]
    )
    M, _ = cv2.estimateAffinePartial2D(src_pts[:3], dst_pts[:3], method=cv2.LMEDS)
    if M is None:
        return edited_bgr, False

    h, w = edited_bgr.shape[:2]
    warped = cv2.warpAffine(
        original_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    alpha = _face_mask((h, w), dst_lm)
    # Full identity lock in the face core; soft blend only at oval edge
    core = (alpha >= 0.55).astype(np.float32)
    edge = np.clip((alpha - 0.15) / 0.40, 0.0, 1.0)
    mix = np.maximum(core, edge * 0.95)
    a = mix[:, :, None]
    out = edited_bgr.astype(np.float32) * (1.0 - a) + warped.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8), True
