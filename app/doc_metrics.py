"""Lightweight product metrics (doc_type counts) for ops grep / Metrika cross-check."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("gosphoto-gate")


def _doc_type_log_path() -> Path:
    # Keep under RESULTS_DIR so the compose volume persists across deploys.
    return config.RESULTS_DIR / "_metrics" / "doc_type.jsonl"


def record_doc_type(
    *,
    doc_type: str,
    result_id: str | None = None,
    dpi: int | None = None,
    source: str = "process",
) -> None:
    """Append one JSONL row + structured log line (safe if disk fails)."""
    row: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "event": "doc_type",
        "doc_type": doc_type,
        "source": source,
    }
    if result_id:
        row["result_id"] = result_id
    if dpi is not None:
        row["dpi"] = int(dpi)
    log.info(
        "DOC_TYPE_METRIC doc_type=%s result_id=%s dpi=%s source=%s",
        doc_type,
        result_id or "-",
        dpi if dpi is not None else "-",
        source,
    )
    try:
        path = _doc_type_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("DOC_TYPE_METRIC write failed: %s", e)
