"""Paste original face/head onto an edited portrait (anti-beautify)."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker

_LEFT_EYE = 33
_RIGHT_EYE = 263
_CHIN = 152
_NOSE = 1

# MediaPipe Face Mesh FACE_OVAL ring (skin contour, not hair/bg).
_FACE_OVAL_IDX = (
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
)


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


def _face_mask(shape: tuple[int, int], pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (face_core, hair_fringe) soft masks. Shoulders stay from the edit."""
    h, w = shape
    face = np.zeros((h, w), np.float32)
    hair = np.zeros((h, w), np.float32)
    if pts.shape[0] >= 468:
        oval = pts[list(_FACE_OVAL_IDX)].astype(np.int32)
        cv2.fillConvexPoly(face, oval, 1.0)
        xs = oval[:, 0].astype(np.float32)
        ys = oval[:, 1].astype(np.float32)
        face_w = float(max(xs.max() - xs.min(), 1.0))
        face_h = float(max(ys.max() - ys.min(), 1.0))
        cx = float(xs.mean())
        crown = float(ys.min())
        chin = float(ys.max())
        # Hair / crown / temples — later gated by person matte
        cv2.ellipse(
            hair,
            (int(cx), int(crown + face_h * 0.05)),
            (max(int(face_w * 0.72), 10), max(int(face_h * 0.55), 10)),
            0,
            180,
            360,
            1.0,
            -1,
        )
        cv2.ellipse(
            hair,
            (int(cx), int((crown + chin) * 0.48)),
            (max(int(face_w * 0.62), 10), max(int(face_h * 0.58), 10)),
            0,
            0,
            360,
            1.0,
            -1,
        )
        hair = np.clip(hair - face, 0.0, 1.0)
        neck_y = int(min(h - 1, chin + face_h * 0.06))
        face[neck_y:, :] = 0.0
        hair[neck_y:, :] = 0.0
    else:
        xs, ys = pts[:, 0], pts[:, 1]
        face_h = float(max(ys.max() - ys.min(), 1.0))
        face_w = float(max(xs.max() - xs.min(), 1.0))
        cx = float(xs.mean())
        cy = float((ys.min() + ys.max()) * 0.50)
        axes = (max(int(face_w * 0.52), 8), max(int(face_h * 0.62), 8))
        cv2.ellipse(face, (int(cx), int(cy)), axes, 0, 0, 360, 1.0, -1)

    blur_f = max(11, (int(0.03 * min(h, w)) | 1))
    blur_h = max(15, (int(0.04 * min(h, w)) | 1))
    face = cv2.GaussianBlur(face, (blur_f, blur_f), 0)
    hair = cv2.GaussianBlur(hair, (blur_h, blur_h), 0)
    return np.clip(face, 0.0, 1.0), np.clip(hair, 0.0, 1.0)


def restore_face_from_original(
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    *,
    include_hair: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Warp original face onto edited image so identity stays natural.

    Shoulders / clothing / bg from `edited_bgr`; face (optional hair) from `original_bgr`.
    Returns (image, applied).
    """
    src_lm = _landmarks_xy(original_bgr)
    dst_lm = _landmarks_xy(edited_bgr)
    if src_lm is None or dst_lm is None:
        return edited_bgr, False

    # Align by eyes only — chin from generative edit often drifts and stretches face
    src_pts = np.float32(
        [
            src_lm[_LEFT_EYE],
            src_lm[_RIGHT_EYE],
            src_lm[_NOSE],
        ]
    )
    dst_pts = np.float32(
        [
            dst_lm[_LEFT_EYE],
            dst_lm[_RIGHT_EYE],
            dst_lm[_NOSE],
        ]
    )
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
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
    face_m, hair_m = _face_mask((original_bgr.shape[0], original_bgr.shape[1]), src_lm)
    if include_hair:
        try:
            from .bg import _onnx_confidence

            conf = _onnx_confidence(original_bgr, "silueta")
            person = (conf >= 0.48).astype(np.float32)
            person = cv2.GaussianBlur(person, (5, 5), 0)
            hair_m = hair_m * np.clip(person, 0.0, 1.0)
        except Exception:
            pass
        src_mask = np.clip(np.maximum(face_m, hair_m), 0.0, 1.0)
    else:
        # Face oval only — hair/shoulders/bg stay from the edited frame (OR)
        src_mask = face_m

    alpha = cv2.warpAffine(
        src_mask,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = np.clip(alpha, 0.0, 1.0)

    face_w = cv2.warpAffine(
        face_m, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    # Hard identity lock — no seamlessClone (it recolors face toward OR beautify)
    mix = np.where(face_w >= 0.28, 1.0, np.clip(alpha / 0.28, 0.0, 1.0))
    a = mix[:, :, None]
    out = edited_bgr.astype(np.float32) * (1.0 - a) + warped.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8), True
