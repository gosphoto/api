"""Persist gate-rejected uploads for later review."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config

log = logging.getLogger("gosphoto-gate")

_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


def _safe_stem(name: str | None) -> str:
    raw = (name or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = Path(raw).stem or "upload"
    cleaned = _SAFE.sub("_", stem).strip("._")[:60]
    return cleaned or "upload"


def _ext_for(data: bytes, filename: str | None) -> str:
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


def save_rejected(
    data: bytes,
    *,
    reason: str | None,
    message: str | None = None,
    metrics: dict | None = None,
    filename: str | None = None,
) -> Path | None:
    """
    Write rejected upload bytes (+ sidecar JSON) into REJECTEDS_DIR.

    Returns the image path, or None if disabled / empty / IO error.
    """
    if not config.REJECTEDS_ENABLED or not data:
        return None
    try:
        root = Path(config.REJECTEDS_DIR)
        root.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        reason_s = _SAFE.sub("_", (reason or "unknown"))[:40] or "unknown"
        stem = f"{ts}_{reason_s}_{_safe_stem(filename)}"
        ext = _ext_for(data, filename)
        img_path = root / f"{stem}{ext}"
        # Avoid rare collisions in the same second
        if img_path.exists():
            img_path = root / f"{stem}_{len(data)}{ext}"
        img_path.write_bytes(data)
        meta = {
            "saved_at": ts,
            "reason": reason,
            "message": message,
            "metrics": metrics or {},
            "original_filename": filename,
            "bytes": len(data),
            "image": img_path.name,
        }
        img_path.with_suffix(img_path.suffix + ".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Saved rejected upload reason=%s path=%s", reason, img_path)
        return img_path
    except Exception as e:
        log.warning("Failed to save rejected upload: %s", e)
        return None
