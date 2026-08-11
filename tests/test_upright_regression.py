"""Regression: inverted selfie must be auto-uprighted before edit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.gate import _analyze_bgr, _decode_image, upright_image

FIX = Path(__file__).resolve().parent / "fixtures" / "upright_regression"


def _cases() -> list[Path]:
    manifest = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
    return [FIX / name for name in manifest["cases"]]


@pytest.mark.parametrize("case_dir", _cases(), ids=lambda p: p.name)
def test_upright_regression_case(case_dir: Path):
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    expect = case["expect"]
    src = case_dir / case.get("input", "in.jpg")
    assert src.is_file(), f"missing {src}"

    data = src.read_bytes()
    oriented, meta = upright_image(data)
    assert oriented is not None
    assert meta.get("rotation_deg") == expect["rotation_deg"], meta

    if "chin_below_forehead" in expect:
        assert bool(meta.get("chin_below_forehead")) is bool(
            expect["chin_below_forehead"]
        ), meta

    if "face_center_y_max" in expect:
        fcy = float(meta["face_center_y"])
        assert fcy <= float(expect["face_center_y_max"]), (
            f"{case['id']}: face_center_y={fcy} > {expect['face_center_y_max']}"
        )

    # Oriented pixels must keep a single face with chin below forehead.
    bgr = _decode_image(oriented)
    assert bgr is not None
    face_count, _yaw, _pitch, _roll, _blur, upright, face_cy = _analyze_bgr(bgr)
    assert face_count == 1
    assert upright is True
    assert face_cy <= float(expect.get("face_center_y_max", 0.6))
