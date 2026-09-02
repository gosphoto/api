"""Regression: bald + high-hair + hijab crop cases. Run when changing crop/compliance."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from app.crop import run_crop_stage

FIX = Path(__file__).resolve().parent / "fixtures" / "crop_regression"


def _cases() -> list[Path]:
    manifest = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
    return [FIX / name for name in manifest["cases"]]


@pytest.mark.parametrize("case_dir", _cases(), ids=lambda p: p.name)
def test_crop_regression_case(case_dir: Path):
    case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    expect = case["expect"]
    src = case_dir / case.get("input", "white_in.jpg")
    assert src.is_file(), f"missing {src}"

    bgr = cv2.imread(str(src))
    assert bgr is not None, f"unreadable {src}"

    cropped, metrics, comp = run_crop_stage(bgr)

    assert cropped is not None
    assert comp.get("pass") is expect.get("pass", True), (
        f"{case['id']}: pass={comp.get('pass')} checks={comp.get('checks')} "
        f"top={comp.get('top_margin')} face={comp.get('face_ratio')}"
    )

    tm = float(comp["top_margin"])
    assert expect["top_margin_min"] <= tm <= expect["top_margin_max"], (
        f"{case['id']}: top_margin={tm} outside "
        f"[{expect['top_margin_min']}, {expect['top_margin_max']}]"
    )

    fr = float(comp["face_ratio"])
    assert fr >= expect["face_ratio_min"], (
        f"{case['id']}: face_ratio={fr} < {expect['face_ratio_min']}"
    )
    if "face_ratio_max" in expect:
        assert fr <= expect["face_ratio_max"], (
            f"{case['id']}: face_ratio={fr} > {expect['face_ratio_max']}"
        )

    hh = float(comp["head_height_mm"])
    assert expect["head_height_mm_min"] <= hh <= expect["head_height_mm_max"], (
        f"{case['id']}: head_height_mm={hh}"
    )

    if "face_only_max" in expect:
        fo = float(comp.get("face_only") or 0)
        assert fo <= float(expect["face_only_max"]), (
            f"{case['id']}: face_only={fo} > {expect['face_only_max']}"
        )

    bald = comp.get("baldness") or metrics.get("baldness")
    if bald is not None and "is_bald" in expect:
        assert bool(bald.get("is_bald")) is bool(expect["is_bald"]), (
            f"{case['id']}: is_bald={bald.get('is_bald')} expected {expect['is_bald']} "
            f"({bald})"
        )
