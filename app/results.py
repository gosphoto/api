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
from .preview import make_preview_jpeg

log = logging.getLogger("gosphoto-gate")

RESULT_ID_RE = re.compile(r"^[a-f0-9]{32}$")
ALLOWED_FILES = frozenset(
    {
        "digital.jpg",
        "print.jpg",
        "preview_digital.jpg",
        "preview_print.jpg",
        "resume.jpg",
        "preview_resume.jpg",
    }
)


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
    resume_jpeg: bytes | None = None,
) -> str | None:
    """
    Write results/<id>/{digital,print,preview_*}.jpg + optional resume + meta.json.

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
        preview_digital = make_preview_jpeg(digital_jpeg)
        preview_print = make_preview_jpeg(print_jpeg)
        (folder / "preview_digital.jpg").write_bytes(preview_digital)
        (folder / "preview_print.jpg").write_bytes(preview_print)

        resume_offer = bool(resume_jpeg)
        if resume_jpeg:
            (folder / "resume.jpg").write_bytes(resume_jpeg)
            (folder / "preview_resume.jpg").write_bytes(make_preview_jpeg(resume_jpeg))

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "result_id": rid,
            "saved_at": ts,
            "digital": "digital.jpg",
            "print": "print.jpg",
            "preview_digital": "preview_digital.jpg",
            "preview_print": "preview_print.jpg",
            "digital_bytes": len(digital_jpeg),
            "print_bytes": len(print_jpeg),
            **(meta or {}),
            "paid": False,
            "resume_offer": resume_offer,
            "paid_resume": False,
            "torso_ok": bool((meta or {}).get("torso_ok", resume_offer)),
        }
        if resume_offer:
            payload["resume"] = "resume.jpg"
            payload["preview_resume"] = "preview_resume.jpg"
            payload["resume_bytes"] = len(resume_jpeg or b"")
        (folder / "meta.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info(
            "Saved result id=%s dir=%s resume_offer=%s",
            rid,
            folder,
            resume_offer,
        )
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


def _write_meta(result_id: str, meta: dict[str, Any]) -> None:
    path = result_dir(result_id) / "meta.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def is_paid(result_id: str) -> bool:
    meta = load_meta(result_id)
    if not meta:
        return False
    return bool(meta.get("paid"))


def is_paid_resume(result_id: str) -> bool:
    meta = load_meta(result_id)
    if not meta:
        return False
    return bool(meta.get("paid_resume"))


def has_resume_offer(result_id: str) -> bool:
    meta = load_meta(result_id)
    if not meta:
        return False
    return bool(meta.get("resume_offer"))


def set_paid(
    result_id: str,
    *,
    payment_id: str,
    tochka_operation_id: str,
    paid_at: str | None = None,
) -> bool:
    meta = load_meta(result_id)
    if not meta:
        return False
    ts = paid_at or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta["paid"] = True
    meta["paid_at"] = ts
    meta["payment_id"] = payment_id
    meta["tochka_operation_id"] = tochka_operation_id
    try:
        _write_meta(result_id, meta)
        return True
    except Exception as e:
        log.warning("Failed to set paid id=%s: %s", result_id, e)
        return False


def set_paid_resume(
    result_id: str,
    *,
    payment_id: str,
    tochka_operation_id: str,
    paid_at: str | None = None,
) -> bool:
    meta = load_meta(result_id)
    if not meta:
        return False
    if not meta.get("resume_offer"):
        log.warning("set_paid_resume without resume_offer id=%s", result_id)
    ts = paid_at or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta["paid_resume"] = True
    meta["paid_resume_at"] = ts
    meta["resume_payment_id"] = payment_id
    meta["resume_tochka_operation_id"] = tochka_operation_id
    try:
        _write_meta(result_id, meta)
        return True
    except Exception as e:
        log.warning("Failed to set paid_resume id=%s: %s", result_id, e)
        return False


def load_file(result_id: str, name: str) -> bytes | None:
    if name not in ALLOWED_FILES:
        return None
    if not is_valid_result_id(result_id):
        return None
    if name in ("preview_digital.jpg", "preview_print.jpg", "preview_resume.jpg"):
        # Always rebuild so old watermarked caches drop after deploy.
        rebuilt = ensure_preview(result_id, name)
        if rebuilt:
            return rebuilt
    path = result_dir(result_id) / name
    if not path.is_file():
        return None
    try:
        return path.read_bytes()
    except Exception as e:
        log.warning("Failed to read result file id=%s name=%s: %s", result_id, name, e)
        return None


def ensure_preview(result_id: str, preview_name: str) -> bytes | None:
    """Rebuild downscaled preview from the full JPEG (overwrites cache)."""
    source_map = {
        "preview_digital.jpg": "digital.jpg",
        "preview_print.jpg": "print.jpg",
        "preview_resume.jpg": "resume.jpg",
    }
    source_name = source_map.get(preview_name)
    if not source_name:
        return None
    folder = result_dir(result_id)
    source = folder / source_name
    if not source.is_file():
        return None
    try:
        preview = make_preview_jpeg(source.read_bytes())
        (folder / preview_name).write_bytes(preview)
        return preview
    except Exception as e:
        log.warning("Failed to build preview id=%s name=%s: %s", result_id, preview_name, e)
        return None
