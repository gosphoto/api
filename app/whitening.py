"""Force studio backgrounds to pure #FFFFFF for Gosuslugi."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker


def _subject_mask(bgr: np.ndarray) -> np.ndarray | None:
    """Protect face + hair + shoulders from bleaching."""
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

    face_h = float(max(ys.max() - ys.min(), 1.0))
    face_w = float(max(xs.max() - xs.min(), 1.0))
    cx = float(xs.mean())
    chin = int(ys.max())
    crown = int(ys.min())

    # Hair dome / temples — light hair often sits outside the landmark hull
    hair_rx = max(int(face_w * 1.15), 10)
    hair_ry = max(int(face_h * 0.85), 10)
    cv2.ellipse(mask, (int(cx), crown), (hair_rx, hair_ry), 0, 180, 360, 255, -1)
    # Full head oval covering ears + side hair
    cv2.ellipse(
        mask,
        (int(cx), int((crown + chin) / 2)),
        (max(int(face_w * 0.95), 10), max(int(face_h * 0.85), 10)),
        0,
        0,
        360,
        255,
        -1,
    )

    k = max(17, (int(0.065 * min(h, w)) | 1))
    mask = cv2.dilate(mask, np.ones((k, k), np.uint8), iterations=2)
    # Soft edge so whitening doesn't leave hard rectangular notches
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.5)
    _, mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)

    cv2.rectangle(
        mask,
        (max(0, int(cx - face_w * 1.1)), max(0, chin - 8)),
        (min(w - 1, int(cx + face_w * 1.1)), h - 1),
        255,
        -1,
    )
    # Do NOT punch corners out of the subject mask — on a tight 35×45 crop
    # shoulders/clothes often reach the bottom corners.
    return mask


def defringe_near_white(
    bgr: np.ndarray,
    *,
    luma_min: int = 200,
    chroma_max: float = 18.0,
) -> np.ndarray:
    """Bleach near-white tinted pixels (classic cutout fringe) to #FFFFFF."""
    if bgr.size == 0:
        return bgr
    out = bgr.copy()
    f = bgr.astype(np.float32)
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_c = f.mean(axis=2, keepdims=True)
    chroma = np.linalg.norm(f - mean_c, axis=2)
    mask = (luma >= float(luma_min)) & (chroma >= 1.0) & (
        chroma <= float(chroma_max) * 3
    )
    near = (luma >= float(luma_min)) & (np.min(f, axis=2) < 250)
    mask = mask | near
    out[mask] = 255
    return out


def _bleach_corner_chips(
    bgr: np.ndarray,
    subject: np.ndarray,
    n: int | None = None,
) -> np.ndarray:
    """Whiten background-like corner pixels only — never paint over subject/clothes."""
    h, w = bgr.shape[:2]
    if n is None:
        n = max(10, int(0.03 * min(h, w)))
    out = bgr.copy()
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    regions = (
        (slice(0, n), slice(0, n)),
        (slice(0, n), slice(w - n, w)),
        (slice(h - n, h), slice(0, n)),
        (slice(h - n, h), slice(w - n, w)),
    )
    for ys, xs in regions:
        bleach = (subject[ys, xs] == 0) & (luma[ys, xs] >= 180)
        chip = out[ys, xs]
        chip[bleach] = 255
        out[ys, xs] = chip
    return out


def force_white_background(bgr: np.ndarray, tol: int = 52) -> np.ndarray:
    """Whiten bg-like pixels outside the subject; soft-clean corners."""
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return bgr

    subject = _subject_mask(bgr)
    if subject is None:
        subject = np.zeros((h, w), np.uint8)

    band = max(8, int(0.05 * min(h, w)))
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

    # Color-similar light pixels outside subject
    bg = ((dist <= float(tol)) & (luma >= 130) & (subject == 0)).astype(np.uint8) * 255

    # Also flood from corners through that candidate set (fills soft bg holes)
    work = (bg > 0).astype(np.uint8)
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = work
    seeds = [(1, 1), (w, 1), (1, h), (w, h)]
    for sx, sy in seeds:
        img = padded.copy()
        mask = np.zeros((h + 4, w + 4), np.uint8)
        if img[sy, sx] == 0:
            img[sy, sx] = 1
        cv2.floodFill(img, mask, (sx, sy), 1, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
        bg[mask[2 : h + 2, 2 : w + 2] > 0] = 255

    bg[subject > 0] = 0
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bg = cv2.dilate(bg, np.ones((3, 3), np.uint8), iterations=1)
    bg[subject > 0] = 0

    alpha = cv2.GaussianBlur(bg, (5, 5), 0).astype(np.float32) / 255.0
    alpha[subject > 0] = 0.0

    out = bgr.astype(np.float32)
    out = out * (1.0 - alpha[:, :, None]) + 255.0 * alpha[:, :, None]

    out[subject > 0] = bgr[subject > 0]
    cleaned = defringe_near_white(np.clip(out, 0, 255).astype(np.uint8))
    protect = subject > 0
    cleaned[protect] = bgr[protect]
    cleaned = _bleach_corner_chips(cleaned, subject)
    cleaned[protect] = bgr[protect]
    return cleaned


def corner_whiteness(bgr: np.ndarray) -> dict:
    """Score background whiteness; skip clothing-dominated bottom corners."""
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
        "white_ok": bool(np.all(avg >= 245)),
    }
