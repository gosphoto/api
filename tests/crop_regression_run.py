"""Print crop metrics for fixtures/crop_regression (manual smoke)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

# Allow `python -m tests.crop_regression_run` from backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.crop import run_crop_stage  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "crop_regression"


def main() -> int:
    manifest = json.loads((FIX / "manifest.json").read_text(encoding="utf-8"))
    failed = 0
    for name in manifest["cases"]:
        case_dir = FIX / name
        case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        src = case_dir / case.get("input", "white_in.jpg")
        bgr = cv2.imread(str(src))
        if bgr is None:
            print(f"{name}: UNREADABLE {src}")
            failed += 1
            continue
        _, metrics, comp = run_crop_stage(bgr)
        exp = case["expect"]
        tm = float(comp["top_margin"]) if comp.get("top_margin") is not None else -1.0
        ok_tm = exp["top_margin_min"] <= tm <= exp["top_margin_max"]
        ok_pass = bool(comp.get("pass")) is bool(exp.get("pass", True))
        status = "OK" if ok_tm and ok_pass else "FAIL"
        if status == "FAIL":
            failed += 1
        bald = comp.get("baldness") or metrics.get("baldness") or {}
        print(
            f"{status}  {name:16}  top={tm:.3f}  face={comp.get('face_ratio')}  "
            f"only={comp.get('face_only')}  "
            f"hh={comp.get('head_height_mm')}mm  pass={comp.get('pass')}  "
            f"bald={bald.get('is_bald')}  crown_f={metrics.get('crown_factor')}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
