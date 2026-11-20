"""Narrow bangs smoke scope: one brow-covering fringe + one clear brows."""

from __future__ import annotations

from pathlib import Path

from app.gate import validate_image

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "bangs_scope"
WITH_BANGS = FIXTURES / "with_bangs_img4761.jpg"
NO_BANGS = FIXTURES / "no_bangs_bgsub.jpg"

BANGS_MESSAGE_SNIPPET = "чёлка закрывает брови"


def test_bangs_scope_fixtures_exist():
    assert WITH_BANGS.is_file(), WITH_BANGS
    assert NO_BANGS.is_file(), NO_BANGS


def test_no_bangs_bgsub_not_rejected_for_bangs():
    gate = validate_image(NO_BANGS.read_bytes())
    assert gate.reason != "bangs_on_brows"
    bangs = (gate.metrics or {}).get("bangs")
    if bangs is not None:
        assert bangs.get("bangs_on_brows") is False


def test_with_bangs_img4761_is_rejected():
    gate = validate_image(WITH_BANGS.read_bytes())
    assert gate.ok is False
    assert gate.reason == "bangs_on_brows"
    assert BANGS_MESSAGE_SNIPPET in gate.message
    bangs = (gate.metrics or {}).get("bangs") or {}
    assert bangs.get("bangs_on_brows") is True
    assert gate.bisenet_mask_png
    assert gate.bisenet_overlay_png
