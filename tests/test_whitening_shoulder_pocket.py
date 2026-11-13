"""Bleach enclosed gray wall pockets between hair and shoulders.

Live miss: gosphoto.ru/result/f3965beb999f8baaa028dcd859b61e63
pair 20261109T070051Z_IMG_4564 — Gemini 2.5 left studio gray in the
hair↔blazer cave; force_white_background kept it under the subject mask.

Corpus: tests/fixtures/whitening/shoulder_pockets/ (MANIFEST.json).
Each case remembers the gray wall color before cleaning and asserts the
cleaner leaves those pixels near #FFFFFF.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.whitening import _shoulder_hair_pocket_mask, force_white_background

FIX = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "whitening"
    / "shoulder_hair_gray_pocket.jpg"
)
NECK_WARM = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "whitening"
    / "neck_warm_wall_pocket.jpg"
)
CROP = Path(__file__).resolve().parent / "fixtures" / "crop_regression"
CORPUS = Path(__file__).resolve().parent / "fixtures" / "whitening" / "shoulder_pockets"
MANIFEST = CORPUS / "MANIFEST.json"

_WALL_LUMA_MIN = 155.0
_WALL_LUMA_MAX = 246.0
_WALL_CHROMA_MAX = 40.0  # warm-tinted neck wall (not only neutral gray)
_MASK_MIN_PX = 800
_AFTER_WHITE_MIN = 245


def _load_manifest() -> list[dict]:
    raw = json.loads(MANIFEST.read_text())
    assert isinstance(raw, list) and raw, "shoulder_pockets MANIFEST empty"
    return raw


def _corpus_pairs() -> list[str]:
    return [str(h["pair"]) for h in _load_manifest()]


def _edit_path(pair: str) -> Path:
    return CORPUS / pair / "edit_gemini.jpg"


def _wall_mask(bgr: np.ndarray) -> np.ndarray:
    return _shoulder_hair_pocket_mask(bgr)


def _wall_color(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean BGR of remembered wall pixels (the gray studio remnant)."""
    assert bool(mask.any())
    return bgr[mask].astype(np.float32).mean(axis=0)


def _assert_wall_color_is_gray_studio(color_bgr: np.ndarray) -> None:
    b, g, r = (float(x) for x in color_bgr)
    luma = (b + g + r) / 3.0
    chroma = max(b, g, r) - min(b, g, r)
    assert _WALL_LUMA_MIN <= luma <= _WALL_LUMA_MAX, (
        f"wall luma out of studio range: {luma:.1f} bgr=({b:.1f},{g:.1f},{r:.1f})"
    )
    assert chroma <= _WALL_CHROMA_MAX, (
        f"wall chroma too high for gray studio: {chroma:.1f}"
    )


def _assert_former_wall_is_white(out: np.ndarray, before_mask: np.ndarray) -> None:
    # Soft AA + JPEG leave a thin rim; score a deeper pocket core.
    core = cv2.erode(before_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if int(core.sum()) < 80:
        core = cv2.erode(before_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if int(core.sum()) < 80:
        core = before_mask
    pix = out[core]
    assert pix.size > 0
    mins = pix.min(axis=1)
    white_frac = float((mins >= _AFTER_WHITE_MIN).mean())
    # 0.93: corpus selfie 20261107T095144Z lands ~0.94 with 3×3 core (CI flake).
    assert white_frac >= 0.93, (
        f"wall remnant after clean: white_frac={white_frac:.3f} "
        f"mean={pix.mean(axis=0)} min_channel_mean={float(mins.mean()):.1f}"
    )


def _left_shoulder_gray_count(bgr: np.ndarray) -> int:
    h, w = bgr.shape[:2]
    y0, y1 = int(h * 0.48), int(h * 0.78)
    x0, x1 = int(w * 0.12), int(w * 0.42)
    roi = bgr[y0:y1, x0:x1].astype(np.float32)
    luma = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    chroma = np.linalg.norm(roi - roi.mean(axis=2, keepdims=True), axis=2)
    gray = (luma >= 165) & (luma <= 245) & (chroma < 28)
    return int(gray.sum())


def test_fixture_has_enclosed_shoulder_gray_pocket():
    bgr = cv2.imread(str(FIX))
    assert bgr is not None
    n = _left_shoulder_gray_count(bgr)
    assert n > 1500, f"fixture lost the live miss pocket: gray={n}"


def test_shoulder_hair_pocket_bleaches_to_white():
    bgr = cv2.imread(str(FIX))
    assert bgr is not None
    before = _left_shoulder_gray_count(bgr)
    out = force_white_background(bgr, tol=48)
    after = _left_shoulder_gray_count(out)
    assert after < 400, f"pocket still gray: before={before} after={after}"
    dark = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 70
    assert int(dark.sum()) > 500
    assert float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[dark].mean()) < 90


def test_does_not_bleach_high_hair_girl_locks():
    path = CROP / "high_hair_girl" / "white_in.jpg"
    bgr = cv2.imread(str(path))
    assert bgr is not None
    out = force_white_background(bgr, tol=55)
    h, w = bgr.shape[:2]
    y0, y1 = int(0.20 * h), int(0.55 * h)
    x0, x1 = int(0.05 * w), int(0.22 * w)
    before = float(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).mean())
    after = float(cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).mean())
    assert before < 200
    assert after < before + 25, f"hair locks bleached: before={before:.1f} after={after:.1f}"


def _side_neck_gray_count(bgr: np.ndarray, side: str) -> int:
    h, w = bgr.shape[:2]
    y0, y1 = int(0.35 * h), int(0.65 * h)
    if side == "L":
        x0, x1 = int(0.15 * w), int(0.35 * w)
    else:
        x0, x1 = int(0.65 * w), int(0.85 * w)
    roi = bgr[y0:y1, x0:x1].astype(np.float32)
    luma = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    chroma = np.linalg.norm(roi - roi.mean(axis=2, keepdims=True), axis=2)
    return int(((luma >= 160) & (luma <= 245) & (chroma < 28)).sum())


def test_neck_warm_wall_strip_bleaches():
    """Warm-tinted gray strip beside the neck (not neutral) must go to white."""
    bgr = cv2.imread(str(NECK_WARM))
    assert bgr is not None
    before_l = _side_neck_gray_count(bgr, "L")
    before_r = _side_neck_gray_count(bgr, "R")
    assert before_l > 2000 and before_r > 2000, (
        f"fixture lost neck wall: L={before_l} R={before_r}"
    )
    mask = _wall_mask(bgr)
    color = _wall_color(bgr, mask)
    _assert_wall_color_is_gray_studio(color)
    out = force_white_background(bgr, tol=48)
    after_l = _side_neck_gray_count(out, "L")
    after_r = _side_neck_gray_count(out, "R")
    assert after_l < 400, f"left neck wall remains: {before_l} -> {after_l}"
    assert after_r < 500, f"right neck wall remains: {before_r} -> {after_r}"
    _assert_former_wall_is_white(out, mask)


def test_pocket_mask_drops_sub_diameter_filaments():
    """Open must remove hair-gap filaments; fat pocket core must survive."""
    bgr = cv2.imread(str(NECK_WARM))
    assert bgr is not None
    mask = _wall_mask(bgr)
    assert int(mask.sum()) > 5000
    # Surviving mask must have a real inscribed radius (not only 1px snakes).
    radius = float(
        cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5).max()
    )
    assert radius >= 5.0, f"pocket core missing after diameter open: r={radius:.2f}"
    # Erode by 3×3 — if only filaments remained, this would wipe the mask.
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    assert int(eroded.sum()) > 2000, "mask is filament-only after diameter limit"


def test_pocket_bleach_has_feathered_not_binary_edge():
    """Pocket whitening must ramp into leftover light wall — not a binary cliff."""
    from app.whitening import _bleach_mask_soft

    bgr = cv2.imread(str(NECK_WARM))
    assert bgr is not None
    mask = _wall_mask(bgr)
    assert int(mask.sum()) > 1000
    soft = _bleach_mask_soft(bgr, mask)
    # Rim = light studio pixels just outside the hard mask.
    dil = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    b_ch, g_ch, r_ch = (c.astype(np.float32) for c in cv2.split(bgr))
    luma = (b_ch + g_ch + r_ch) / 3.0
    chroma = np.maximum(np.maximum(b_ch, g_ch), r_ch) - np.minimum(
        np.minimum(b_ch, g_ch), r_ch
    )
    rim = dil & ~mask & (luma >= 155) & (luma <= 250) & (chroma < 42)
    assert int(rim.sum()) > 30
    binary_dil = bgr.copy()
    binary_dil[dil] = 255
    soft_l = float(soft[rim].astype(np.float32).mean())
    src_l = float(bgr[rim].astype(np.float32).mean())
    bin_l = float(binary_dil[rim].astype(np.float32).mean())
    assert soft_l > src_l + 8, "feather did not lift light rim toward white"
    assert soft_l < bin_l - 2, "rim is a hard binary step, not a ramp"


@pytest.mark.parametrize("pair", _corpus_pairs())
def test_corpus_has_gray_wall_before(pair: str):
    path = _edit_path(pair)
    assert path.is_file(), f"missing edit_gemini for {pair}"
    bgr = cv2.imread(str(path))
    assert bgr is not None
    mask = _wall_mask(bgr)
    n = int(mask.sum())
    assert n >= _MASK_MIN_PX, f"{pair}: wall remnants missing (mask_px={n})"
    color = _wall_color(bgr, mask)
    _assert_wall_color_is_gray_studio(color)


@pytest.mark.parametrize("pair", _corpus_pairs())
def test_corpus_cleaner_removes_gray_wall(pair: str):
    path = _edit_path(pair)
    bgr = cv2.imread(str(path))
    assert bgr is not None
    before_mask = _wall_mask(bgr)
    assert int(before_mask.sum()) >= _MASK_MIN_PX
    wall_color = _wall_color(bgr, before_mask)
    _assert_wall_color_is_gray_studio(wall_color)

    out = force_white_background(bgr, tol=48)
    _assert_former_wall_is_white(out, before_mask)

    # Dark clothing / hair core must survive.
    dark = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) < 70
    if int(dark.sum()) > 200:
        assert float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[dark].mean()) < 100
