"""After passport payment, render the full frame from the original pair.

The webhook does not wait. Resume payments and a repeated mark_paid do not start this.
After Riverflow Pro, also crop+print a Gemini variant from archived edit_gemini.jpg.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from . import results
from .pairs import (
    EDIT_GEMINI,
    EDIT_RIVERFLOW,
    OUT_RIVERFLOW,
    resolve_pair_dir,
    save_pair_file,
)

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


def _decode_jpeg_bgr(data: bytes) -> np.ndarray | None:
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return bgr


def _build_gemini_variant(result_id: str, meta: dict) -> bool:
    """Crop+print from archived edit_gemini.jpg (no OpenRouter call)."""
    from .crop import encode_jpeg, run_crop_stage
    from .main import _print_payload

    pair_dir = resolve_pair_dir(meta.get("source_pair"))
    if pair_dir is None:
        log.info("Paid gemini variant skip id=%s reason=pair_dir_missing", result_id)
        return False
    edit_path = pair_dir / EDIT_GEMINI
    if not edit_path.is_file():
        log.info("Paid gemini variant skip id=%s reason=edit_gemini_missing", result_id)
        return False
    try:
        edit_bytes = edit_path.read_bytes()
    except OSError as e:
        log.warning("Paid gemini variant read failed id=%s: %s", result_id, e)
        return False
    bgr = _decode_jpeg_bgr(edit_bytes)
    if bgr is None:
        log.warning("Paid gemini variant decode failed id=%s", result_id)
        return False
    preset = config.resolve_doc_preset(meta.get("doc_type"))
    out_w = int(preset["width"])
    out_h = int(preset["height"])
    dpi = int(preset["dpi"])
    jpeg_max = int(preset["jpeg_max_bytes"])
    try:
        cropped, crop_metrics, compliance = run_crop_stage(
            bgr, width=out_w, height=out_h, dpi=dpi
        )
        jpeg = encode_jpeg(cropped, max_bytes=jpeg_max, dpi=dpi)
        print_jpeg, print_sheet = _print_payload(cropped, include_base64=False)
    except Exception as e:
        log.warning("Paid gemini variant crop failed id=%s: %s", result_id, e)
        return False
    compliance = {
        **compliance,
        "jpeg_bytes": len(jpeg),
        "jpeg_max_bytes": jpeg_max,
        "jpeg_size_ok": len(jpeg) <= jpeg_max,
    }
    print_meta = {
        k: print_sheet[k]
        for k in ("width", "height", "dpi", "copies", "bytes", "size_cm", "mime", "layout")
        if k in print_sheet
    }
    ok = results.attach_variant_jpegs(
        result_id,
        results.VARIANT_GEMINI,
        jpeg,
        print_jpeg,
        crop=crop_metrics,
        compliance=compliance,
        print_sheet=print_meta,
        primary_alias=False,
        mark_ready=False,
    )
    if ok:
        log.info(
            "Paid gemini variant ready id=%s digital_bytes=%s print_bytes=%s",
            result_id,
            len(jpeg),
            len(print_jpeg),
        )
    return ok


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
    # Best-effort archive only — never fail the paid frame on disk I/O.
    edited_jpeg = out.get("edited_jpeg") or b""
    digital_jpeg = out.get("jpeg") or b""
    edit_meta = out.get("edit_meta") if isinstance(out.get("edit_meta"), dict) else {}
    source_pair = meta.get("source_pair")
    model_name = edit_meta.get("model") or config.RIVERFLOW_PRO_MODEL
    log.info(
        "Paid riverflow archive start id=%s source_pair=%s edit_bytes=%s out_bytes=%s model=%s",
        result_id,
        source_pair,
        len(edited_jpeg),
        len(digital_jpeg),
        model_name,
    )
    try:
        pair_dir = resolve_pair_dir(source_pair)
        if pair_dir is None:
            log.warning(
                "Paid riverflow archive skip id=%s reason=pair_dir_missing source_pair=%s",
                result_id,
                source_pair,
            )
        else:
            if edited_jpeg:
                saved_edit = save_pair_file(
                    pair_dir,
                    edited_jpeg,
                    name=EDIT_RIVERFLOW,
                    model=model_name,
                )
                log.info(
                    "Paid edit_riverflow id=%s ok=%s bytes=%s",
                    result_id,
                    saved_edit is not None,
                    len(edited_jpeg),
                )
            else:
                log.warning(
                    "Paid edit_riverflow skip id=%s reason=empty_jpeg", result_id
                )
            if digital_jpeg:
                saved_out = save_pair_file(
                    pair_dir,
                    digital_jpeg,
                    name=OUT_RIVERFLOW,
                    model=model_name,
                )
                log.info(
                    "Paid out_riverflow id=%s ok=%s bytes=%s",
                    result_id,
                    saved_out is not None,
                    len(digital_jpeg),
                )
            else:
                log.warning(
                    "Paid out_riverflow skip id=%s reason=empty_jpeg", result_id
                )
    except Exception as e:
        log.warning("Paid riverflow archive failed id=%s: %s", result_id, e)
    ok = results.attach_full_jpegs(
        result_id,
        out["jpeg"],
        out["print_jpeg"],
        crop=out.get("crop_metrics"),
        compliance=out.get("compliance"),
        print_sheet=out.get("print_meta"),
        edit=out.get("edit_meta"),
        mark_ready=False,
    )
    if not ok:
        log.warning("Paid attach_full_jpegs failed id=%s", result_id)
        _set_status(result_id, "error")
        return
    # Second variant from preview edit archive (CPU crop only).
    meta_after = results.load_meta(result_id) or meta
    try:
        _build_gemini_variant(result_id, meta_after)
    except Exception as e:
        log.warning("Paid gemini variant unexpected id=%s: %s", result_id, e)
    _set_status(result_id, "ready")
    log.info("Paid full frame ready id=%s", result_id)


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
