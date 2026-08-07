"""Face-protect model: MediaPipe Face Mesh no-retouch zone.

Inside this zone only original selfie pixels are allowed — no generative
rewrite, no CLAHE/beautify, no whitening bleed.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker

_LEFT_EYE = 33
_RIGHT_EYE = 263

# MediaPipe FACE_OVAL — skin contour only (not hair / room bg).
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


def _similarity_from_eyes(src_lm: np.ndarray, dst_lm: np.ndarray) -> np.ndarray | None:
    """Map src eye line → dst eye line (scale+rotate+translate). Scale clamped."""
    s0 = src_lm[_LEFT_EYE].astype(np.float64)
    s1 = src_lm[_RIGHT_EYE].astype(np.float64)
    d0 = dst_lm[_LEFT_EYE].astype(np.float64)
    d1 = dst_lm[_RIGHT_EYE].astype(np.float64)
    s_vec, d_vec = s1 - s0, d1 - d0
    s_len, d_len = float(np.linalg.norm(s_vec)), float(np.linalg.norm(d_vec))
    if s_len < 1e-3 or d_len < 1e-3:
        return None
    scale = float(np.clip(d_len / s_len, 0.85, 1.18))
    ang = float(np.arctan2(d_vec[1], d_vec[0]) - np.arctan2(s_vec[1], s_vec[0]))
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    t = (d0 + d1) * 0.5 - R @ ((s0 + s1) * 0.5)
    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = R.astype(np.float32)
    M[:, 2] = t.astype(np.float32)
    return M


def face_protect_mask_from_landmarks(
    shape: tuple[int, int], lm: np.ndarray
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    if lm.shape[0] < 468:
        xs, ys = lm[:, 0], lm[:, 1]
        cx, cy = float(xs.mean()), float((ys.min() + ys.max()) * 0.5)
        axes = (
            max(int((xs.max() - xs.min()) * 0.52), 8),
            max(int((ys.max() - ys.min()) * 0.62), 8),
        )
        cv2.ellipse(mask, (int(cx), int(cy)), axes, 0, 0, 360, 1.0, -1)
    else:
        oval = lm[list(_FACE_OVAL_IDX)].astype(np.int32)
        cv2.fillConvexPoly(mask, oval, 1.0)
        xs = oval[:, 0].astype(np.float32)
        ys = oval[:, 1].astype(np.float32)
        face_w = float(max(xs.max() - xs.min(), 1.0))
        face_h = float(max(ys.max() - ys.min(), 1.0))
        cx = float(xs.mean())
        crown = float(ys.min())
        cv2.ellipse(
            mask,
            (int(cx), int(crown + face_h * 0.06)),
            (max(int(face_w * 0.40), 8), max(int(face_h * 0.18), 6)),
            0,
            180,
            360,
            1.0,
            -1,
        )
        neck_y = int(min(h - 1, ys.max() + face_h * 0.02))
        mask[neck_y:, :] = 0.0

    blur = max(11, (int(0.03 * min(h, w)) | 1))
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return np.clip(mask, 0.0, 1.0)


def face_protect_mask(bgr: np.ndarray) -> np.ndarray | None:
    lm = _landmarks_xy(bgr)
    if lm is None:
        return None
    return face_protect_mask_from_landmarks(bgr.shape[:2], lm)


def align_edit_to_original(
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """
    Warp the edited frame so its eyes land on the original eye positions.

    Keeps the original face grid fixed; moves OR bg/shoulders underneath.
    """
    src_lm = _landmarks_xy(original_bgr)
    ed_lm = _landmarks_xy(edited_bgr)
    if src_lm is None or ed_lm is None:
        if edited_bgr.shape[:2] != original_bgr.shape[:2]:
            return (
                cv2.resize(
                    edited_bgr,
                    (original_bgr.shape[1], original_bgr.shape[0]),
                    interpolation=cv2.INTER_AREA,
                ),
                False,
            )
        return edited_bgr, False

    h, w = original_bgr.shape[:2]
    # Map edited → original (inverse of "paste face onto OR")
    M = _similarity_from_eyes(ed_lm, src_lm)
    if M is None:
        return (
            cv2.resize(edited_bgr, (w, h), interpolation=cv2.INTER_AREA)
            if edited_bgr.shape[:2] != (h, w)
            else edited_bgr
        ), False

    aligned = cv2.warpAffine(
        edited_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, True


def apply_face_protect(
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
) -> tuple[np.ndarray, bool, dict]:
    """
    1) Align edit eyes → original eyes (move OR, not the face).
    2) Paste original face in-place (no face warp → no 'съехало').
    """
    h, w = original_bgr.shape[:2]
    if original_bgr.shape[:2] != edited_bgr.shape[:2]:
        # Will be aligned to original size below
        pass

    aligned, did_align = align_edit_to_original(original_bgr, edited_bgr)
    lm = _landmarks_xy(original_bgr)
    if lm is None:
        return aligned, False, {"model": "mediapipe_face_mesh", "applied": False}

    alpha = face_protect_mask_from_landmarks((h, w), lm)
    mix = np.where(alpha >= 0.30, 1.0, np.clip(alpha / 0.30, 0.0, 1.0))
    a = mix[:, :, None]
    out = aligned.astype(np.float32) * (1.0 - a) + original_bgr.astype(np.float32) * a
    meta = {
        "model": "mediapipe_face_mesh",
        "applied": True,
        "align": "edit_to_original_eyes" if did_align else "resize_only",
        "protect_coverage": round(float((mix >= 0.99).mean()), 4),
    }
    return np.clip(out, 0, 255).astype(np.uint8), True, meta


def protect_from_retouch(bgr: np.ndarray, original_bgr: np.ndarray) -> np.ndarray:
    out, ok, _ = apply_face_protect(original_bgr, bgr)
    return out if ok else bgr
