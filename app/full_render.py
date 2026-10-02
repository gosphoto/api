"""After passport payment, render the full frame from the original pair.

The webhook does not wait. Resume payments and a repeated mark_paid do not start this.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from . import config
from . import results

log = logging.getLogger("gosphoto-gate")

_lock = threading.Lock()
_inflight: set[str] = set()


def _stored_status(meta: dict) -> str | None:
    full = meta.get("full")
    if isinstance(full, dict):
        status = full.get("status")
        return status if isinstance(status, str) else None
    if isinstance(full, str):
        return full
    return None


def files_ready(result_id: str) -> bool:
    folder = results.result_dir(result_id)
    return (folder / "digital.jpg").is_file() and (folder / "print.jpg").is_file()


def public_status(result_id: str, meta: dict) -> str | None:
    """Status the page may show. None until the passport is paid."""
    if not meta.get("paid"):
        return None
    if files_ready(result_id):
        return "ready"
    if _stored_status(meta) == "error":
        return "error"
    return "pending"


def _set_status(result_id: str, status: str) -> None:
    results.set_full_status(result_id, status)


def load_source_bytes(meta: dict) -> bytes | None:
    rel = meta.get("source_pair")
    if not isinstance(rel, str) or not rel or rel.startswith(("/", "\\")) or ".." in rel:
        return None
    root = Path(config.PAIRS_DIR).resolve()
    folder = (root / rel).resolve()
    if folder != root and root not in folder.parents:
        return None
    if not folder.is_dir():
        return None
    matches = sorted(p for p in folder.glob("in.*") if p.is_file())
    if not matches:
        return None
    try:
        return matches[0].read_bytes()
    except OSError as e:
        log.warning("Failed to read pair source %s: %s", matches[0], e)
        return None


def _passport_stages(data: bytes, *, mime: str, preset: dict, edit_model: str):
    from .main import _run_passport_stages

    return _run_passport_stages(
        data,
        mime=mime,
        preset=preset,
        edit_model=edit_model,
    )


def _generate(result_id: str) -> None:
    if files_ready(result_id):
        _set_status(result_id, "ready")
        return
    meta = results.load_meta(result_id) or {}
    data = load_source_bytes(meta)
    if not data:
        log.warning("full render missing source id=%s", result_id)
        _set_status(result_id, "error")
        return
    preset = config.resolve_doc_preset(meta.get("doc_type"))
    out = _passport_stages(
        data,
        mime="image/jpeg",
        preset=preset,
        edit_model=config.RIVERFLOW_PRO_MODEL,
    )
    if not out.get("ok"):
        log.warning(
            "full render failed id=%s stage=%s message=%s",
            result_id,
            out.get("stage"),
            out.get("message"),
        )
        _set_status(result_id, "error")
        return
    ok = results.attach_full_jpegs(
        result_id,
        out["jpeg"],
        out["print_jpeg"],
        crop=out.get("crop_metrics"),
        compliance=out.get("compliance"),
        print_sheet=out.get("print_meta"),
        edit=out.get("edit_meta"),
    )
    if not ok:
        _set_status(result_id, "error")


def _run(result_id: str) -> None:
    try:
        _generate(result_id)
    except Exception:
        log.exception("full render crashed id=%s", result_id)
        _set_status(result_id, "error")
    finally:
        with _lock:
            _inflight.discard(result_id)


def ensure_started(result_id: str, *, retry_error: bool = False) -> None:
    """Start the paid full frame once. A ready result is left alone."""
    if not results.is_valid_result_id(result_id):
        return
    meta = results.load_meta(result_id)
    if not meta or not meta.get("paid"):
        return
    if files_ready(result_id):
        if _stored_status(meta) != "ready":
            _set_status(result_id, "ready")
        return
    stored = _stored_status(meta)
    with _lock:
        if result_id in _inflight:
            return
        if stored == "error" and not retry_error:
            return
        _inflight.add(result_id)
    _set_status(result_id, "pending")
    threading.Thread(
        target=_run,
        args=(result_id,),
        name=f"full-{result_id[:8]}",
        daemon=True,
    ).start()


def wait_until_ready(result_id: str, timeout_sec: float = 180) -> str:
    """Block until the full frame is ready, failed, or the wait runs out."""
    ensure_started(result_id, retry_error=False)
    deadline = time.monotonic() + timeout_sec
    while True:
        meta = results.load_meta(result_id) or {}
        status = public_status(result_id, meta)
        if status in ("ready", "error"):
            return status
        if time.monotonic() >= deadline:
            return status or "pending"
        time.sleep(0.4)
