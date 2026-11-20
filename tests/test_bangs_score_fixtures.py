"""BiSeNet bangs score on bangs_gate JPGs (no MediaPipe)."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.bangs import bangs_on_brows_score

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bangs_gate"


def _jpgs(split: str) -> list[Path]:
    return sorted((FIXTURES / split).glob("*.jpg"))


@pytest.mark.parametrize("path", _jpgs("positive"), ids=lambda p: p.name)
def test_positive_bangs_score_hits(path: Path):
    bgr = cv2.imread(str(path))
    assert bgr is not None
    scores = bangs_on_brows_score(bgr)
    assert scores.get("bangs_on_brows") is True, (path.name, scores)
    assert scores.get("bisenet_mask_png")
    assert scores.get("bisenet_overlay_png")


@pytest.mark.parametrize("path", _jpgs("negative"), ids=lambda p: p.name)
def test_negative_bangs_score_misses(path: Path):
    bgr = cv2.imread(str(path))
    assert bgr is not None
    scores = bangs_on_brows_score(bgr)
    assert scores.get("bangs_on_brows") is False, (path.name, scores)
