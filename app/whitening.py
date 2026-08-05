"""Force studio backgrounds to pure #FFFFFF for Gosuslugi."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker


def _tight_face_mask(bgr: np.ndarray) -> np.ndarray | None:
    """Tight head+shoulders mask — used only to avoid bleaching skin/hair."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    xs = np.array([p.x * w for p in lm], dtype=np.float32)
    ys = np.array([p.y * h for p in lm], dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    # hair above forehead + cheeks margin (keep away from image corners)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    # clear outer frame so corners stay bleachable
    edge = max(8, int(0.05 * min(h, w)))
    mask[:edge, :] = 0
    mask[-edge:, :] = 0
    mask[:, :edge] = 0
    mask[:, -edge:] = 0
    chin = int(ys.max())
    cx = float(xs.mean())
    face_w = float(xs.max() - xs.min())
    cv2.rectangle(
        mask,
        (max(edge, int(cx - face_w * 0.9)), max(edge, chin - 5)),
        (min(w - 1 - edge, int(cx + face_w * 0.9)), h - 1 - edge),
        255,
        -1,
    )
    return mask


def force_white_background(bgr: np.ndarray, tol: int = 55) -> np.ndarray:
    """Whiten by color (border-like pixels), never paint the face hull."""
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return bgr

    face = _tight_face_mask(bgr)
    if face is None:
        face = np.zeros((h, w), np.uint8)

    band = max(8, int(0.05 * min(h, w)))
    # Prefer top corners for bg color (less likely to hit shoulders)
    sample = np.concatenate(
        [
            bgr[:band, :band].reshape(-1, 3),
            bgr[:band, -band:].reshape(-1, 3),
            bgr[:band, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    med = np.median(sample, axis=0)

    dist = np.linalg.norm(bgr.astype(np.float32) - med, axis=2)
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    bg = (dist <= float(tol)) & (luma >= 120) & (face == 0)
    bg_u8 = (bg.astype(np.uint8) * 255)
    bg_u8 = cv2.morphologyEx(bg_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bg_u8 = cv2.dilate(bg_u8, np.ones((3, 3), np.uint8), iterations=2)
    bg_u8[face > 0] = 0

    # Outer frame always white (except face overlap)
    bg_u8[:band, :] = 255
    bg_u8[-band:, :] = 255
    bg_u8[:, :band] = 255
    bg_u8[:, -band:] = 255
    bg_u8[face > 0] = 0

    alpha = cv2.GaussianBlur(bg_u8, (5, 5), 0).astype(np.float32) / 255.0
    alpha[face > 0] = 0.0
    # hard-set corners to pure white regardless of alpha blur
    out = bgr.astype(np.float32)
    white = np.full_like(out, 255.0)
    alpha3 = alpha[:, :, None]
    out = out * (1.0 - alpha3) + white * alpha3
    out[:band, :] = 255
    out[-band:, :] = 255
    out[:, :band] = 255
    out[:, -band:] = 255
    # restore face (but never inside the white border band)
    face_bool = face > 0
    face_bool[:band, :] = False
    face_bool[-band:, :] = False
    face_bool[:, :band] = False
    face_bool[:, -band:] = False
    out[face_bool] = bgr[face_bool]
    return np.clip(out, 0, 255).astype(np.uint8)


def corner_whiteness(bgr: np.ndarray) -> dict:
    n = 12
    corners = [
        bgr[:n, :n],
        bgr[:n, -n:],
        bgr[-n:, :n],
        bgr[-n:, -n:],
    ]
    means = [c.reshape(-1, 3).mean(axis=0) for c in corners]
    avg = np.mean(means, axis=0)
    return {
        "bgr": [round(float(x), 1) for x in avg],
        "white_ok": bool(np.all(avg >= 245)),
    }
