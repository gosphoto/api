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
CROP = Path(__file__).resolve().parent / "fixtures" / "crop_regression"
CORPUS = Path(__file__).resolve().parent / "fixtures" / "whitening" / "shoulder_pockets"
MANIFEST = CORPUS / "MANIFEST.json"

_WALL_LUMA_MIN = 155.0
_WALL_LUMA_MAX = 246.0
_WALL_CHROMA_MAX = 32.0
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
    pix = out[before_mask]
    assert pix.size > 0
    mins = pix.min(axis=1)
    white_frac = float((mins >= _AFTER_WHITE_MIN).mean())
    assert white_frac >= 0.95, (
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
