"""Hard-reject uploads where bangs cover the eyebrows (Gosuslugi face oval)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.gate import validate_image

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bangs_gate"
POSITIVE = sorted((FIXTURES / "positive").glob("*.jpg"))
NEGATIVE = sorted((FIXTURES / "negative").glob("*.jpg"))

BANGS_MESSAGE_SNIPPET = "чёлка закрывает брови"


def test_bangs_fixture_counts():
    assert len(POSITIVE) >= 30, f"need ≥30 positive fixtures, got {len(POSITIVE)}"
    assert len(NEGATIVE) >= 30, f"need ≥30 negative fixtures, got {len(NEGATIVE)}"
    assert (FIXTURES / "positive" / "semenova.jpg").is_file()


@pytest.mark.parametrize("path", POSITIVE, ids=lambda p: p.name)
def test_bangs_on_brows_photos_are_rejected(path: Path):
    gate = validate_image(path.read_bytes())
    assert gate.ok is False
    assert gate.reason == "bangs_on_brows"
    assert BANGS_MESSAGE_SNIPPET in gate.message
    bangs = (gate.metrics or {}).get("bangs") or {}
    assert bangs.get("bangs_on_brows") is True


@pytest.mark.parametrize("path", NEGATIVE, ids=lambda p: p.name)
def test_no_bangs_photos_are_not_rejected_for_bangs(path: Path):
    gate = validate_image(path.read_bytes())
    assert gate.reason != "bangs_on_brows"
    bangs = (gate.metrics or {}).get("bangs")
    if bangs is not None:
        assert bangs.get("bangs_on_brows") is False
