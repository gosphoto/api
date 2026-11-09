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
    ring_px = max(
        1,
        int(round(float(os.getenv("HAIR_EDGE_SOFTEN_RING_PX", "3")) * _HAIR_SOFTEN_THIN)),
    )
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


def _is_hair_wall_spill_case(bgr: np.ndarray, chin_y: int | None = None) -> bool:
    """True only for leftover-wall halo around hair (her Gosuslugi miss)."""
    if bgr.size == 0 or min(bgr.shape[:2]) < 8:
        return False
    cool = _hair_wall_spill_mask(bgr, chin_y=chin_y)
    n = int(cool.sum())
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


# 30% thinner than the first production edge (ring 3px, blur σ 1.5 / 0.8).
_HAIR_SOFTEN_THIN = 0.7


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
    color_sigma = float(os.getenv("HAIR_EDGE_SOFTEN_COLOR_SIGMA", "1.5")) * _HAIR_SOFTEN_THIN
    band_sigma = 1.5 * _HAIR_SOFTEN_THIN
    src = np.clip(out, 0, 255).astype(np.uint8)
    smoothed = cv2.GaussianBlur(src, (0, 0), color_sigma)
    smoothed = cv2.GaussianBlur(smoothed, (0, 0), color_sigma * 0.65)
    mix_color = cv2.GaussianBlur(band.astype(np.float32), (0, 0), band_sigma)
    mix_color = np.clip(mix_color * 0.85, 0.0, 1.0)
    out = out * (1.0 - mix_color[:, :, None]) + smoothed.astype(np.float32) * mix_color[
        :, :, None
    ]

    # Alpha feather on core — narrow anti-alias (~5px), low white pull.
    sigma = float(os.getenv("HAIR_EDGE_SOFTEN_SIGMA", "0.8")) * _HAIR_SOFTEN_THIN
    alpha = cv2.GaussianBlur(core_u8.astype(np.float32) / 255.0, (0, 0), sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    lighten = float(os.getenv("HAIR_EDGE_SOFTEN_LIGHTEN", "0.05"))
    bg_layer = out * (1.0 - lighten) + 255.0 * lighten
    feathered = out * alpha[:, :, None] + bg_layer * (1.0 - alpha[:, :, None])

    mix = cv2.GaussianBlur(band.astype(np.float32), (0, 0), band_sigma)
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


def _shoulder_hair_pocket_mask(bgr: np.ndarray) -> np.ndarray:
    """Enclosed mid-gray studio crumbs between hair and shoulders.

    Gemini often leaves the original wall in hair↔jacket caves — including a
    warm-tinted gray strip beside the neck that a fat face ellipse used to
    swallow. Bleach lateral flat crumbs that touch #FFFFFF; refuse only deep
    textured blonde locks far from the white plate.

    Cavity width is limited with morphological opening (no grow-into-wall):
    thin filaments between strands are dropped so bleach does not punch holes
    in the hair silhouette.
    """
    h, w = bgr.shape[:2]
    if min(h, w) < 32:
        return np.zeros((h, w), dtype=bool)

    geom = _face_geom(bgr)
    if geom is None:
        return np.zeros((h, w), dtype=bool)
    chin_y, crown_y, cx, face_w = geom

    b_ch, g_ch, r_ch = (c.astype(np.float32) for c in cv2.split(bgr))
    luma = (b_ch + g_ch + r_ch) / 3.0
    chroma = np.maximum(np.maximum(b_ch, g_ch), r_ch) - np.minimum(
        np.minimum(b_ch, g_ch), r_ch
    )
    blur = cv2.GaussianBlur(luma, (5, 5), 0)
    local_std = np.sqrt(np.maximum(cv2.blur((luma - blur) ** 2, (5, 5)), 0.0))

    y0 = int(crown_y + 0.08 * (chin_y - crown_y))
    y1 = min(h, chin_y + int(face_w * 1.55))
    band = np.zeros((h, w), dtype=bool)
    band[y0:y1, :] = True

    yy, xx = np.ogrid[:h, :w]
    # Tight face oval — neck-side wall must stay outside.
    face = (
        (xx - cx) ** 2 / max(face_w * 0.48, 1.0) ** 2
        + (yy - (crown_y + chin_y) / 2.0) ** 2
        / max((chin_y - crown_y) * 0.58, 1.0) ** 2
    ) <= 1.0

    hair = _head_hair_region(
        chin_y=chin_y, crown_y=crown_y, cx=cx, face_w=face_w, h=h, w=w
    )
    white = luma >= 248.0
    dist = cv2.distanceTransform((~white).astype(np.uint8), cv2.DIST_L2, 5)
    warm_deep = (
        hair
        & (r_ch > b_ch + 14)
        & (chroma > 18)
        & (luma >= 145.0)
        & (luma <= 230.0)
        & (dist > 16)
        & (local_std > 10)
    )
    near_white = cv2.dilate(white.astype(np.uint8), np.ones((19, 19), np.uint8)) > 0

    wall = (
        band
        & (~face)
        & (~warm_deep)
        & (luma >= 140.0)
        & (luma <= 246.0)
        & (chroma < 38.0)
    )
    # Warm-tinted gray strip beside the neck (live miss 1000323483 / screenshot).
    neck_y0 = int(crown_y + 0.45 * (chin_y - crown_y))
    neck = (
        (yy >= neck_y0)
        & (yy <= chin_y + int(face_w * 0.85))
        & (~face)
        & (np.abs(xx - cx) > face_w * 0.32)
        & (luma >= 155.0)
        & (luma <= 238.0)
        & (chroma < 45.0)
        & near_white
        & (local_std <= 20.0)
    )
    wall = wall | neck

    closed = cv2.morphologyEx(
        wall.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), np.uint8),
    )
    num, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros((h, w), dtype=bool)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 15 or area > 100000:
            continue
        m = labels == i
        if not bool(near_white[m].any()):
            continue
        if float(chroma[m].mean()) > 36.0:
            continue
        xs = np.where(m)[1]
        if xs.size == 0:
            continue
        if abs(float(np.median(xs)) - cx) < face_w * 0.20:
            continue
        keep |= m

    keep &= wall
    if not bool(keep.any()):
        return keep
    # Do NOT grow into `wall` — that crawls between hair strands and punches
    # holes in the silhouette. Limit cavity diameter with morphological open:
    # filaments thinner than open_px are dropped; fat neck/shoulder pockets stay.
    # Adaptive: fat pockets (inscribed radius ≥ 8) use 5px; thin real wall strips
    # keep a milder 3px open so they still bleach.
    pocket_r = float(
        cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5).max()
    )
    open_px = (5 if pocket_r >= 8.0 else 3) | 1
    opened = cv2.morphologyEx(
        keep.astype(np.uint8) * 255,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)),
    )
    return opened > 0


def _bleach_mask_soft(
    bgr: np.ndarray,
    mask: np.ndarray,
    *,
    sigma: float | None = None,
) -> np.ndarray:
    """Paint pocket white with a soft ramp into leftover light studio gray.

    Feather only into already-light wall-like pixels (never into dark hair —
    that created mid-gray halos). Then anti-alias the white seam so the cut
    is not stair-stepped; restore solid white on the original mask after.
    """
    if not bool(mask.any()):
        return bgr
    h, w = bgr.shape[:2]
    m = float(min(h, w))
    if sigma is None:
        sigma = max(2.8, 0.0055 * m)
    core_u8 = (mask.astype(np.uint8)) * 255
    b_ch, g_ch, r_ch = (c.astype(np.float32) for c in cv2.split(bgr))
    luma = (b_ch + g_ch + r_ch) / 3.0
    chroma = np.maximum(np.maximum(b_ch, g_ch), r_ch) - np.minimum(
        np.minimum(b_ch, g_ch), r_ch
    )
    light_wall = (luma >= 155.0) & (luma <= 250.0) & (chroma < 42.0)

    dist = cv2.distanceTransform(core_u8, cv2.DIST_L2, 5)
    solid = dist >= 1.5
    out = bgr.copy()
    out[solid] = 255
    # Soft ramp for the mask rim + a halo into leftover light studio gray.
    rim = cv2.dilate(core_u8, np.ones((5, 5), np.uint8), iterations=2)
    field = cv2.GaussianBlur(rim, (0, 0), sigmaX=float(sigma)).astype(np.float32) / 255.0
    # Near-full on every mask pixel so wall clears; solid core stays pure white.
    alpha = np.maximum(field, mask.astype(np.float32) * 0.96)
    alpha = np.maximum(alpha, solid.astype(np.float32))
    paint = (~solid) & light_wall & (alpha > 0.04)
    if bool(paint.any()) or bool((mask & ~solid).any()):
        src = bgr.astype(np.float32)
        blended = src * (1.0 - alpha[:, :, None]) + 255.0 * alpha[:, :, None]
        use = (mask & ~solid) | paint
        out = np.where(use[:, :, None], blended, out.astype(np.float32))
        out = np.clip(out, 0, 255).astype(np.uint8)
    # Anti-alias stair-steps along the bright seam; keep solid core pure.
    grad = cv2.morphologyEx(
        core_u8,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    seam = cv2.GaussianBlur(
        grad, (0, 0), sigmaX=max(2.0, 0.0045 * m)
    ).astype(np.float32) / 255.0
    out_luma = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mix = (seam > 0.04) & (out_luma >= 190.0) & (~solid)
    if bool(mix.any()):
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=max(2.4, 0.005 * m))
        s = np.clip(seam * 0.8, 0.0, 1.0)[:, :, None]
        mixed = out.astype(np.float32) * (1.0 - s) + blurred.astype(np.float32) * s
        out = np.where(mix[:, :, None], mixed, out.astype(np.float32))
        out = np.clip(out, 0, 255).astype(np.uint8)
    out[solid] = 255
    return out


def _bleach_shoulder_hair_pockets(
    bgr: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Feather studio pockets to #FFFFFF. No-op when none fire."""
    if mask is None:
        mask = _shoulder_hair_pocket_mask(bgr)
    return _bleach_mask_soft(bgr, mask)


def force_white_background(
    bgr: np.ndarray,
    tol: int = 52,
    *,
    soften: bool = True,
) -> np.ndarray:
    """Whiten bg-like pixels; optionally run final hair/background softening."""
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
    cleaned = defringe_near_white(np.clip(out, 0, 255).astype(np.uint8))
    cleaned[hard] = bgr[hard]
    cleaned = _bleach_corner_chips(cleaned, np.where(hard, 255, 0).astype(np.uint8))
    cleaned[hard] = bgr[hard]
    # After hard restore: kill enclosed hair↔shoulder gray Gemini leaves behind.
    # Mask on the *source* (pre-whiten): hard restore puts those wall crumbs
    # back; bleaching only on `cleaned` would miss pockets that the plate
    # already partially ate or reshaped. Soft feather — hard paint left tooth
    # edges against hair.
    src_pocket = _shoulder_hair_pocket_mask(bgr)
    cleaned = _bleach_shoulder_hair_pockets(cleaned, mask=src_pocket)
    # Residual crumbs after plate reshape: hard fill only (no seam AA) so we
    # do not re-gray the already-feathered source pocket.
    residual = _shoulder_hair_pocket_mask(cleaned)
    if bool(residual.any()):
        cleaned = cleaned.copy()
        cleaned[residual] = 255
    # Defer both soft passes until after the optional post-crop model cleanup.
    if soften:
        cleaned = _blur_hair_wall_spill(cleaned, chin_y=chin_y)
    if soften and _hair_edge_soften_enabled():
        cleaned = soften_hair_edge_and_bg(
            cleaned,
            chin_y=chin_y,
            subject_hard=np.where(hard, 255, 0).astype(np.uint8),
        )
    # Soft blur can re-gray pocket cores — restore solid white, keep feathered rim.
    if bool(src_pocket.any()):
        solid = (
            cv2.distanceTransform(src_pocket.astype(np.uint8), cv2.DIST_L2, 5) >= 1.5
        )
        if bool(solid.any()):
            cleaned = cleaned.copy()
            cleaned[solid] = 255
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
