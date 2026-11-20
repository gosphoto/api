"""TDD: BiSeNet hair ∩ face-oval on bangs_gate overlay masks."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.bangs import hair_oval_intersection

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bangs_gate"
OVERLAYS = FIXTURES / "overlays"


def _mask_paths(split: str) -> list[Path]:
    root = OVERLAYS / split
    if not root.is_dir():
        return []
    return sorted(root.glob("*.mask.png"))


def _load_mask(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert mask is not None, path
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


@pytest.mark.skipif(
    not _mask_paths("positive") or not _mask_paths("negative"),
    reason="bangs_gate overlays missing — run scripts/bangs_gate_bisenet_overlays.py",
)
def test_overlay_corpus_size():
    assert len(_mask_paths("positive")) >= 30
    assert len(_mask_paths("negative")) >= 30


@pytest.mark.parametrize(
    "path",
    _mask_paths("negative") or [pytest.param(None, id="no-neg-overlays")],
    ids=lambda p: p.name if p else "missing",
)
def test_negative_no_hair_oval_hit(path: Path | None):
    if path is None:
        pytest.skip("overlays not generated")
    scores = hair_oval_intersection(_load_mask(path))
    assert scores["hit"] is False, (path.name, scores)
    assert scores["bangs_on_brows"] is False
    for key in ("l_brow", "r_brow", "l_eye", "r_eye", "nose", "mouth"):
        assert key in scores
        assert "ratio" in scores[key]


@pytest.mark.parametrize(
    "path",
    _mask_paths("positive") or [pytest.param(None, id="no-pos-overlays")],
    ids=lambda p: p.name if p else "missing",
)
def test_positive_hair_oval_hit(path: Path | None):
    if path is None:
        pytest.skip("overlays not generated")
    scores = hair_oval_intersection(_load_mask(path))
    assert scores["hit"] is True, (path.name, scores)
    assert scores["bangs_on_brows"] is True
    for key in ("l_brow", "r_brow", "l_eye", "r_eye", "nose", "mouth"):
        assert key in scores
        assert "ratio" in scores[key]
