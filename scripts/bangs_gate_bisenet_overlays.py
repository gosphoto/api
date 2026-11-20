#!/usr/bin/env python3
"""Generate BiSeNet overlay+mask for bangs_gate fixtures (TDD corpus)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from app import face_parse as fp  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "bangs_gate"
OUT = FIXTURES / "overlays"


def _run_split(split: str) -> int:
    src = FIXTURES / split
    dest = OUT / split
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.glob("*.jpg")):
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f"SKIP unreadable {path.name}", flush=True)
            continue
        art = fp.parse_with_artifacts(bgr)
        stem = path.stem
        (dest / f"{stem}.mask.png").write_bytes(art["mask_png"])
        (dest / f"{stem}.overlay.png").write_bytes(art["overlay_png"])
        print(f"OK {split}/{stem}", flush=True)
        n += 1
    return n


def main() -> int:
    total = 0
    for split in ("positive", "negative"):
        total += _run_split(split)
    print(f"done overlays={total}", flush=True)
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
