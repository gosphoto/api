"""Force studio backgrounds to pure #FFFFFF for Gosuslugi."""

from __future__ import annotations

import os

import cv2
import mediapipe as mp
import numpy as np

from .gate import _landmarker


def _subject_mask(bgr: np.ndarray) -> tuple[np.ndarray | None, int | None]:
    """Protect face + hair + shoulders from bleaching. Chin y, if known."""
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = _landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not result.face_landmarks:
        return None, None
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
    return mask, chin


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


def _studio_plate_mask(
    bgr: np.ndarray,
    subject: np.ndarray,
    *,
    chin_y: int | None,
    luma_min: int = 200,
    chroma_max: float = 24.0,
    std_max: float = 10.0,
) -> np.ndarray:
    """Border-connected flat light fill (Gemini gray studio), even under a fat mask.

    Landmark dilation on a 35×45 crop covers the ~10% top margin. Those pixels
    are still a uniform near-white plate connected to the frame edge — bleach
    them. Do not walk through a shirt island below the chin; gray behind
    the shoulders is the same component as the top plate and must bleach.
    """
    h, w = bgr.shape[:2]
    f = bgr.astype(np.float32)
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    chroma = np.linalg.norm(f - f.mean(axis=2, keepdims=True), axis=2)
    blur = cv2.GaussianBlur(luma, (5, 5), 0)
    local_std = np.sqrt(np.maximum(cv2.blur((luma - blur) ** 2, (5, 5)), 0.0))
    cand = (luma >= float(luma_min)) & (chroma <= chroma_max) & (local_std <= std_max)

    seeds = cand.astype(np.uint8)
    work = np.zeros((h + 2, w + 2), np.uint8)
    work[1:-1, 1:-1] = seeds
    # Seed from the TOP edge only. Bottom-edge seeds would flood a white shirt
    # that touches the 35×45 bottom; left/right full-height seeds would flood a
    # light tee that meets the frame. Gray behind the shoulders is the same
    # connected component as the top plate, so it fills from above.
    work[0, :] = 1
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(
        work,
        mask,
        (0, 0),
        1,
        flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
    )
    plate = (mask[2 : h + 2, 2 : w + 2] > 0).astype(np.uint8) * 255
    plate = cv2.morphologyEx(plate, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return plate


def _y_cut(h: int, chin_y: int | None) -> int:
    y = chin_y if chin_y is not None else int(0.62 * h)
    return max(8, min(h - 1, int(y)))


def _face_geom(bgr: np.ndarray) -> tuple[int, int, float, float] | None:
    """Chin y, crown y, face center x, face width — for hair soften zoning."""
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
    return (
        int(ys.max()),
        int(ys.min()),
        float(xs.mean()),
        float(max(xs.max() - xs.min(), 1.0)),
    )


def _ring_band(mask: np.ndarray, *, px: int = 6) -> np.ndarray:
    """Thin ring around a boolean region."""
    u8 = mask.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    outer = cv2.dilate(u8, k) > 0
    inner = cv2.erode(u8, k) > 0
    return outer & (~inner)


def _head_hair_region(
    *,
    chin_y: int,
    crown_y: int,
    cx: float,
    face_w: float,
    h: int,
    w: int,
) -> np.ndarray:
    """Head ellipse + narrow side locks. Geometry only — never reaches shoulders."""
    region = np.zeros((h, w), bool)
    cy = int(crown_y + 0.52 * (chin_y - crown_y))
    ax = max(int(face_w * 1.24), 12)
    ay = max(int((chin_y - crown_y) * 1.15 + face_w * 0.20), 12)
    head = np.zeros((h, w), np.uint8)
    cv2.ellipse(head, (int(cx), cy), (ax, ay), 0, 0, 360, 255, -1)
    region |= head > 0

    # Clip dome before neck / shoulders.
    y_cap = min(h, chin_y + int(face_w * 0.18))
    region[y_cap:, :] = False

    # Side locks: outer strips only, short — ends before shoulder line.
    y_side_end = min(h, chin_y + int(face_w * 0.32))
    x_in = max(0, int(cx - face_w * 0.50))
    x_out = min(w, int(cx + face_w * 0.50))
    if y_side_end > chin_y:
        region[chin_y:y_side_end, :x_in] = True
        region[chin_y:y_side_end, x_out:] = True

    return region


def _hair_soften_band(
    bgr: np.ndarray,
    *,
    chin_y: int,
    crown_y: int,
    cx: float,
    face_w: float,
) -> np.ndarray:
    """Ring on real head silhouette (flyaways included), only where it meets white."""
    h, w = bgr.shape[:2]
    region = _head_hair_region(
        chin_y=chin_y, crown_y=crown_y, cx=cx, face_w=face_w, h=h, w=w
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    white = gray >= 240
    person = (gray < 240) & region
    person = (
        cv2.morphologyEx(
            person.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )
        > 0
    )
    ring_px = int(os.getenv("HAIR_EDGE_SOFTEN_RING_PX", "3"))
    ring = _ring_band(person, px=ring_px)
    near_white = cv2.dilate(white.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return ring & near_white & region


def _hair_wall_spill_mask(bgr: np.ndarray, *, chin_y: int | None) -> np.ndarray:
    """Cool near-white crumbs sitting on the hair / #FFFFFF boundary.

    Only pixels that touch the white plate *and* dark hair. Interior face,
    bald scalp specks, and shirts below the chin never enter this mask.
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    y_cut = _y_cut(h, chin_y)
    hair = gray < 140
    hair[y_cut:, :] = False
    white = gray >= 250
    k = np.ones((5, 5), np.uint8)
    near_w = cv2.dilate(white.astype(np.uint8), k) > 0
    near_h = cv2.dilate(hair.astype(np.uint8), k) > 0
    band = near_w & near_h & (~hair)
    band[y_cut:, :] = False
    blue, green, red = cv2.split(bgr)
    cool = (
        band
        & (gray >= 210)
        & (gray < 252)
        & (blue.astype(np.int16) + 2 >= red.astype(np.int16))
        & (blue.astype(np.int16) >= green.astype(np.int16) - 2)
    )
    return cool


def measure_outline_symmetry(
    bgr: np.ndarray,
    *,
    gray_thr: int = 200,
    asym_thr_px: int = 3,
) -> float:
    """Max left/right silhouette extent gap in the crown band (pixels)."""
    geom = _face_geom(bgr)
    if geom is None:
        return 0.0
    chin_i, crown_i, cx, face_w = geom
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    region = _head_hair_region(
        chin_y=chin_i, crown_y=crown_i, cx=cx, face_w=face_w, h=h, w=w
    )
    y0 = max(0, crown_i)
    y1 = min(h, crown_i + max(8, int((chin_i - crown_i) * 0.35)))
    max_asym = 0.0
    for y in range(y0, y1):
        row = region[y, :] & (gray[y, :] < gray_thr)
        if not row.any():
            continue
        xs = np.flatnonzero(row)
        left_x, right_x = int(xs.min()), int(xs.max())
        asym = abs((cx - left_x) - (right_x - cx))
        max_asym = max(max_asym, float(asym))
    return max_asym if max_asym > asym_thr_px else 0.0


def _symmetrize_hair_outline(
    bgr: np.ndarray,
    *,
    chin_y: int | None = None,
    cx: float | None = None,
    gray_thr: int = 200,
    asym_thr_px: int = 3,
) -> np.ndarray:
    """Feather a protruding hair side instead of bleaching it away on light walls."""
    geom = _face_geom(bgr)
    h, w = bgr.shape[:2]
    if geom is not None:
        chin_i, crown_i, cx_f, face_w = geom
    elif cx is not None and chin_y is not None:
        chin_i = int(chin_y)
        crown_i = max(8, chin_i - int(h * 0.22))
        cx_f = float(cx)
        face_w = w * 0.35
    else:
        return bgr
    if cx is not None:
        cx_f = float(cx)
    if chin_y is not None:
        chin_i = int(chin_y)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    region = _head_hair_region(
        chin_y=chin_i, crown_y=crown_i, cx=cx_f, face_w=face_w, h=h, w=w
    )
    y0 = max(0, crown_i)
    y1 = min(h, crown_i + max(8, int((chin_i - crown_i) * 0.35)))
    protect = np.zeros((h, w), bool)
    for y in range(y0, y1):
        row = region[y, :] & (gray[y, :] < gray_thr)
        if not row.any():
            continue
        xs = np.flatnonzero(row)
        left_x, right_x = int(xs.min()), int(xs.max())
        left_ext = cx_f - left_x
        right_ext = right_x - cx_f
        if left_ext > right_ext + asym_thr_px:
            protect[y, left_x : int(cx_f)] = True
        elif right_ext > left_ext + asym_thr_px:
            protect[y, int(cx_f) : right_x + 1] = True
    if not protect.any():
        return bgr
    out = bgr.astype(np.float32)
    band = cv2.dilate(protect.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    blurred = cv2.GaussianBlur(bgr, (3, 3), 0)
    mix = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.0) * 0.45
    mix = np.clip(mix, 0.0, 1.0)[:, :, None]
    return np.clip(out * (1.0 - mix) + blurred.astype(np.float32) * mix, 0, 255).astype(
        np.uint8
    )


def _light_bg_portrait(bgr: np.ndarray) -> bool:
    from .bg import _top_corner_luma

    return _top_corner_luma(bgr) >= 185.0


def _light_bg_low_contrast(bgr: np.ndarray) -> bool:
    """Light gray wall (not passport-white studio plate)."""
    cw = corner_whiteness(bgr)
    if cw.get("white_ok"):
        return False
    corner = cw.get("bgr") or [255.0, 255.0, 255.0]
    return float(np.mean(corner)) >= 185.0

def _is_hair_wall_spill_case(bgr: np.ndarray, chin_y: int | None = None) -> bool:
    """True for leftover-wall halo around hair (incl. light studio walls)."""
    if bgr.size == 0 or min(bgr.shape[:2]) < 8:
        return False
    cool = _hair_wall_spill_mask(bgr, chin_y=chin_y)
    n = int(cool.sum())
    if _light_bg_low_contrast(bgr) and n >= 30:
        w = bgr.shape[1]
        if int(cool[:, : w // 2].sum()) >= 10 and int(cool[:, w // 2 :].sum()) >= 10:
            return True
    if n < 100:
        return False
    w = bgr.shape[1]
    return int(cool[:, : w // 2].sum()) >= 20 and int(cool[:, w // 2 :].sum()) >= 20


def _blur_hair_wall_spill(bgr: np.ndarray, chin_y: int | None = None) -> np.ndarray:
    """Soften leftover wall on the hair edge. No-op when the case does not fire."""
    if not _is_hair_wall_spill_case(bgr, chin_y=chin_y):
        return bgr
    h, w = bgr.shape[:2]
    y_cut = _y_cut(h, chin_y)
    cool = _hair_wall_spill_mask(bgr, chin_y=chin_y)
    band = cv2.dilate(cool.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    band[y_cut:, :] = False
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    band[gray < 140] = False
    blurred = cv2.GaussianBlur(bgr, (7, 7), 0)
    mix = cv2.addWeighted(blurred, 0.55, np.full_like(bgr, 255), 0.45, 0)
    out = bgr.copy()
    out[band] = mix[band]
    out[gray < 140] = bgr[gray < 140]
    return out


def _hair_core_mask(gray: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Dark subject core — excludes gray Gemini fringe at the silhouette."""
    core = (gray < 168) & region
    u8 = core.astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return u8 > 0


def soften_hair_edge_and_bg(
    bgr: np.ndarray,
    *,
    chin_y: int | None = None,
    subject_hard: np.ndarray | None = None,
) -> np.ndarray:
    """Anti-alias hair↔#FFFFFF: bleach gray fringe + feather tight silhouette."""
    if bgr.size == 0 or min(bgr.shape[:2]) < 16:
        return bgr

    h, w = bgr.shape[:2]
    geom = _face_geom(bgr)
    if geom is not None:
        chin_i, crown_i, cx, face_w = geom
    else:
        chin_i = _y_cut(h, chin_y)
        crown_i = max(8, chin_i - int(h * 0.22))
        cx, face_w = w / 2.0, w * 0.35

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    region = _head_hair_region(
        chin_y=chin_i, crown_y=crown_i, cx=cx, face_w=face_w, h=h, w=w
    )

    band = _hair_soften_band(
        bgr, chin_y=chin_i, crown_y=crown_i, cx=cx, face_w=face_w
    )

    from .face_protect import face_protect_mask

    fpm = face_protect_mask(bgr)
    if fpm is not None:
        band = band & (fpm < 0.20)

    if int(band.sum()) < 20:
        return bgr

    # Tight dark core → morph open eats 1–2px jaggies before feather.
    if subject_hard is not None and subject_hard.size:
        core = (subject_hard > 0) & region
    else:
        core = (gray < 185) & region
    core_u8 = cv2.morphologyEx(
        core.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    core_u8 = cv2.morphologyEx(core_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    out = bgr.astype(np.float32)

    # Color smooth in band (visible on cutout, no full-frame white smear).
    color_sigma = float(os.getenv("HAIR_EDGE_SOFTEN_COLOR_SIGMA", "1.5"))
    src = np.clip(out, 0, 255).astype(np.uint8)
    smoothed = cv2.GaussianBlur(src, (0, 0), color_sigma)
    smoothed = cv2.GaussianBlur(smoothed, (0, 0), color_sigma * 0.65)
    mix_color = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.5)
    mix_color = np.clip(mix_color * 0.85, 0.0, 1.0)
    out = out * (1.0 - mix_color[:, :, None]) + smoothed.astype(np.float32) * mix_color[
        :, :, None
    ]

    # Alpha feather on core — narrow anti-alias (~5px), low white pull.
    sigma = float(os.getenv("HAIR_EDGE_SOFTEN_SIGMA", "0.8"))
    alpha = cv2.GaussianBlur(core_u8.astype(np.float32) / 255.0, (0, 0), sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    lighten = float(os.getenv("HAIR_EDGE_SOFTEN_LIGHTEN", "0.05"))
    bg_layer = out * (1.0 - lighten) + 255.0 * lighten
    feathered = out * alpha[:, :, None] + bg_layer * (1.0 - alpha[:, :, None])

    mix = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.5)
    mix = np.clip(
        mix * float(os.getenv("HAIR_EDGE_SOFTEN_STRENGTH", "0.35")), 0.0, 1.0
    )
    out = out * (1.0 - mix[:, :, None]) + feathered * mix[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _hair_edge_soften_enabled() -> bool:
    return os.getenv("HAIR_EDGE_SOFTEN", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def force_white_background(bgr: np.ndarray, tol: int = 52) -> np.ndarray:
    """Whiten bg-like pixels outside the subject; soft-clean corners."""
    h, w = bgr.shape[:2]
    if h < 8 or w < 8:
        return bgr

    subject, chin_y = _subject_mask(bgr)
    if subject is None:
        subject = np.zeros((h, w), np.uint8)
        chin_y = None

    plate = _studio_plate_mask(bgr, subject, chin_y=chin_y)
    # Fat hair dilation is not the person — don't restore gray studio over it.
    hard = (subject > 0) & (plate == 0)

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

    # Color-similar light pixels outside the hard subject
    bg = ((dist <= float(tol)) & (luma >= 130) & (~hard)).astype(np.uint8) * 255
    bg = cv2.bitwise_or(bg, plate)

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

    bg[hard] = 0
    bg = cv2.morphologyEx(bg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    bg = cv2.dilate(bg, np.ones((3, 3), np.uint8), iterations=1)
    bg[hard] = 0

    alpha = cv2.GaussianBlur(bg, (5, 5), 0).astype(np.float32) / 255.0
    alpha[hard] = 0.0

    out = bgr.astype(np.float32)
    out = out * (1.0 - alpha[:, :, None]) + 255.0 * alpha[:, :, None]

    out[hard] = bgr[hard]
    light_bg = _light_bg_low_contrast(bgr)
    cleaned = defringe_near_white(
        np.clip(out, 0, 255).astype(np.uint8),
        luma_min=200,
    )
    cleaned[hard] = bgr[hard]
    cleaned = _bleach_corner_chips(cleaned, np.where(hard, 255, 0).astype(np.uint8))
    cleaned[hard] = bgr[hard]
    cleaned = _blur_hair_wall_spill(cleaned, chin_y=chin_y)
    if light_bg and measure_outline_symmetry(cleaned) > 3.0:
        geom = _face_geom(cleaned) or _face_geom(bgr)
        cx = float(geom[2]) if geom is not None else None
        cleaned = _symmetrize_hair_outline(cleaned, chin_y=chin_y, cx=cx)
    if _hair_edge_soften_enabled():
        cleaned = soften_hair_edge_and_bg(
            cleaned,
            chin_y=chin_y,
            subject_hard=np.where(hard, 255, 0).astype(np.uint8),
        )
    return cleaned


_CORNER_WHITE_MIN = 245.0


def corner_whiteness(bgr: np.ndarray) -> dict:
    """Score background whiteness; skip clothing-dominated bottom corners.

    Bottom corners often contain shoulders / a white shirt. Those pixels are
    bright (luma ≥210) but not passport-white — averaging them used to drag a
    clean plate below the ≥245 threshold. Only count bottom pixels that are
    already pure white; otherwise ignore that corner and trust the top ones.
    """
    n = 12
    chips = [
        bgr[:n, :n],
        bgr[:n, -n:],
        bgr[-n:, :n],
        bgr[-n:, -n:],
    ]
    means: list[np.ndarray] = []
    min_keep = max(4, n * n // 4)
    for i, c in enumerate(chips):
        pix = c.reshape(-1, 3).astype(np.float32)
        if i >= 2:
            # Pure white only — not light-gray clothing.
            keep = np.min(pix, axis=1) >= _CORNER_WHITE_MIN
            if int(keep.sum()) < min_keep:
                continue
            pix = pix[keep]
        means.append(pix.mean(axis=0))
    if not means:
        means = [c.reshape(-1, 3).mean(axis=0) for c in chips[:2]]
    avg = np.mean(means, axis=0)
    return {
        "bgr": [round(float(x), 1) for x in avg],
        "white_ok": bool(np.all(avg >= _CORNER_WHITE_MIN)),
    }
