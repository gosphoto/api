"""Persist successful process pairs: input upload + output JPEG."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("gosphoto-gate")

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_stem(name: str | None) -> str:
    raw = (name or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = Path(raw).stem or "upload"
    cleaned = _SAFE.sub("_", stem).strip("._")[:40]
    return cleaned or "upload"


def _in_ext(data: bytes, filename: str | None) -> str:
    name = (filename or "").lower()
    if name.endswith(".png"):
        return ".png"
    if name.endswith(".webp"):
        return ".webp"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def save_pair(
    input_bytes: bytes,
    output_jpeg: bytes,
    *,
    filename: str | None = None,
    meta: dict[str, Any] | None = None,
) -> Path | None:
    """
    Create pairs/<timestamp>_<stem>/ with in.* and out.jpg (+ meta.json).

    Returns the pair folder path, or None if disabled / empty / IO error.
    """
    if not config.PAIRS_ENABLED or not input_bytes or not output_jpeg:
        return None
    try:
        root = Path(config.PAIRS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        folder = root / f"{ts}_{_safe_stem(filename)}"
        if folder.exists():
            folder = root / f"{ts}_{_safe_stem(filename)}_{len(input_bytes)}"
        folder.mkdir(parents=True, exist_ok=False)

        in_path = folder / f"in{_in_ext(input_bytes, filename)}"
        out_path = folder / "out.jpg"
        in_path.write_bytes(input_bytes)
        out_path.write_bytes(output_jpeg)

        payload = {
            "saved_at": ts,
            "original_filename": filename,
            "in": in_path.name,
            "out": out_path.name,
            "in_bytes": len(input_bytes),
            "out_bytes": len(output_jpeg),
            **(meta or {}),
        }
        (folder / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Saved process pair dir=%s", folder)
        return folder
    except Exception as e:
        log.warning("Failed to save process pair: %s", e)
        return None
