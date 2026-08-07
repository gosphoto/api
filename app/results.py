"""Persist shareable process results for /result/<id> pages."""

from __future__ import annotations

import json
import logging
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("gosphoto-gate")

RESULT_ID_RE = re.compile(r"^[a-f0-9]{32}$")


def new_result_id() -> str:
    return secrets.token_hex(16)


def is_valid_result_id(result_id: str) -> bool:
    return bool(RESULT_ID_RE.fullmatch(result_id or ""))


def result_dir(result_id: str) -> Path:
    if not is_valid_result_id(result_id):
        raise ValueError("invalid result id")
    return Path(config.RESULTS_DIR) / result_id


def save_result(
    digital_jpeg: bytes,
    print_jpeg: bytes,
    *,
    meta: dict[str, Any] | None = None,
    result_id: str | None = None,
) -> str | None:
    """
    Write results/<id>/{digital.jpg,print.jpg,meta.json}.

    Returns result_id, or None if disabled / empty / IO error.
    """
    if not config.RESULTS_ENABLED or not digital_jpeg or not print_jpeg:
        return None
    rid = result_id or new_result_id()
    if not is_valid_result_id(rid):
        log.warning("Refusing to save result with invalid id")
        return None
    try:
        folder = result_dir(rid)
        folder.mkdir(parents=True, exist_ok=False)
        (folder / "digital.jpg").write_bytes(digital_jpeg)
        (folder / "print.jpg").write_bytes(print_jpeg)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "result_id": rid,
            "saved_at": ts,
            "digital": "digital.jpg",
            "print": "print.jpg",
            "digital_bytes": len(digital_jpeg),
            "print_bytes": len(print_jpeg),
            **(meta or {}),
        }
        (folder / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Saved result id=%s dir=%s", rid, folder)
        return rid
    except Exception as e:
        log.warning("Failed to save result: %s", e)
        return None


def load_meta(result_id: str) -> dict[str, Any] | None:
    if not is_valid_result_id(result_id):
        return None
    path = result_dir(result_id) / "meta.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read result meta id=%s: %s", result_id, e)
        return None


def load_file(result_id: str, name: str) -> bytes | None:
    if name not in ("digital.jpg", "print.jpg"):
        return None
    if not is_valid_result_id(result_id):
        return None
    path = result_dir(result_id) / name
    if not path.is_file():
        return None
    try:
        return path.read_bytes()
    except Exception as e:
        log.warning("Failed to read result file id=%s name=%s: %s", result_id, name, e)
        return None
