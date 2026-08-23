"""Same-day dedup cache for /api/process by oriented upload bytes + doc_type."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .results import clone_result, is_valid_result_id, result_dir

log = logging.getLogger("gosphoto-gate")


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cache_day_dir() -> Path:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path(config.PROCESS_CACHE_DIR) / day


def _cache_path(digest: str, doc_type: str) -> Path:
    safe_doc = doc_type.replace("/", "_").replace("\\", "_")
    version = config.PROCESS_CACHE_VERSION
    return _cache_day_dir() / f"{digest}_{safe_doc}_v{version}.json"


def lookup(data: bytes, doc_type: str) -> str | None:
    """
    If the same oriented upload was processed today, clone the cached result.

    Returns a fresh unpaid result_id, or None on miss.
    """
    if not config.PROCESS_CACHE_ENABLED or not config.RESULTS_ENABLED:
        return None
    digest = content_hash(data)
    path = _cache_path(digest, doc_type)
    if not path.is_file():
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Process cache read failed path=%s: %s", path, e)
        return None
    source_id = entry.get("result_id")
    if not is_valid_result_id(source_id):
        return None
    src = result_dir(source_id)
    if not (src / "digital.jpg").is_file() or not (src / "print.jpg").is_file():
        log.info("Process cache stale (missing files) source=%s", source_id)
        return None
    cloned = clone_result(source_id)
    if cloned:
        log.info(
            "Process cache hit digest=%s doc_type=%s source=%s clone=%s",
            digest[:12],
            doc_type,
            source_id,
            cloned,
        )
    return cloned


def put(data: bytes, doc_type: str, result_id: str) -> None:
    """Remember the first successful result for this upload today."""
    if not config.PROCESS_CACHE_ENABLED or not config.RESULTS_ENABLED:
        return
    if not is_valid_result_id(result_id):
        return
    digest = content_hash(data)
    path = _cache_path(digest, doc_type)
    if path.is_file():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload = {
            "result_id": result_id,
            "content_hash": digest,
            "doc_type": doc_type,
            "saved_at": ts,
            "cache_version": config.PROCESS_CACHE_VERSION,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
        log.info(
            "Process cache put digest=%s doc_type=%s result_id=%s",
            digest[:12],
            doc_type,
            result_id,
        )
    except Exception as e:
        log.warning("Process cache put failed path=%s: %s", path, e)
