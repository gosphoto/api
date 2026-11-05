"""Persist shareable process results for /result/<id> pages."""

from __future__ import annotations

import json
import logging
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from .preview import make_preview_jpeg, make_view_jpeg

log = logging.getLogger("gosphoto-gate")

RESULT_ID_RE = re.compile(r"^[a-f0-9]{32}$")

VARIANT_RIVERFLOW = "riverflow"
VARIANT_GEMINI = "gemini"
VARIANT_LABELS = {
    VARIANT_RIVERFLOW: "Вариант 1",
    VARIANT_GEMINI: "Вариант 2",
}
VARIANT_FILES = {
    VARIANT_RIVERFLOW: {
        "digital": "digital_riverflow.jpg",
        "print": "print_riverflow.jpg",
        "preview_digital": "preview_digital_riverflow.jpg",
        "preview_print": "preview_print_riverflow.jpg",
    },
    VARIANT_GEMINI: {
        "digital": "digital_gemini.jpg",
        "print": "print_gemini.jpg",
        "preview_digital": "preview_digital_gemini.jpg",
        "preview_print": "preview_print_gemini.jpg",
    },
}

ALLOWED_FILES = frozenset(
    {
        "digital.jpg",
        "print.jpg",
        "preview_digital.jpg",
        "preview_print.jpg",
        "view_digital.jpg",
        "resume.jpg",
        "preview_resume.jpg",
        *(f for spec in VARIANT_FILES.values() for f in spec.values()),
    }
)


def _variant_entry(variant_id: str) -> dict[str, str]:
    files = VARIANT_FILES[variant_id]
    return {
        "id": variant_id,
        "label": VARIANT_LABELS[variant_id],
        "digital": files["digital"],
        "print": files["print"],
        "preview_digital": files["preview_digital"],
        "preview_print": files["preview_print"],
    }


def _upsert_variant(meta: dict[str, Any], entry: dict[str, str]) -> None:
    variants = meta.get("variants")
    if not isinstance(variants, list):
        variants = []
    out = [v for v in variants if isinstance(v, dict) and v.get("id") != entry["id"]]
    # riverflow first, then gemini
    if entry["id"] == VARIANT_RIVERFLOW:
        out.insert(0, entry)
    else:
        out.append(entry)
    # stable order: riverflow then gemini then others
    order = {VARIANT_RIVERFLOW: 0, VARIANT_GEMINI: 1}
    out.sort(key=lambda v: order.get(str(v.get("id")), 99))
    meta["variants"] = out


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


def save_preview_result(
    preview_digital: bytes,
    preview_print: bytes,
    *,
    meta: dict[str, Any] | None = None,
    result_id: str | None = None,
    resume_jpeg: bytes | None = None,
) -> str | None:
    """Write only screen previews. Full digital.jpg / print.jpg come after payment."""
    if not config.RESULTS_ENABLED or not preview_digital or not preview_print:
        return None
    rid = result_id or new_result_id()
    if not is_valid_result_id(rid):
        log.warning("Refusing to save preview result with invalid id")
        return None
    try:
        folder = result_dir(rid)
        folder.mkdir(parents=True, exist_ok=False)
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
            "preview_digital": "preview_digital.jpg",
            "preview_print": "preview_print.jpg",
            "preview_only": True,
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
        log.info("Saved preview result id=%s dir=%s", rid, folder)
        return rid
    except Exception as e:
        log.warning("Failed to save preview result: %s", e)
        return None


def set_full_status(result_id: str, status: str) -> bool:
    meta = load_meta(result_id)
    if not meta:
        return False
    full = meta.get("full") if isinstance(meta.get("full"), dict) else {}
    meta["full"] = {**full, "status": status}
    try:
        _write_meta(result_id, meta)
        return True
    except Exception as e:
        log.warning("Failed to set full status id=%s: %s", result_id, e)
        return False


def attach_variant_jpegs(
    result_id: str,
    variant_id: str,
    digital_jpeg: bytes,
    print_jpeg: bytes,
    *,
    crop: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
    print_sheet: dict[str, Any] | None = None,
    edit: dict[str, Any] | None = None,
    primary_alias: bool = False,
    mark_ready: bool = False,
) -> bool:
    """Write one paid variant (digital + 10×15 + previews) and update meta.variants."""
    if variant_id not in VARIANT_FILES:
        return False
    if not digital_jpeg or not print_jpeg:
        return False
    if not load_meta(result_id):
        return False
    files = VARIANT_FILES[variant_id]
    try:
        folder = result_dir(result_id)
        preview_digital = make_preview_jpeg(digital_jpeg)
        preview_print = make_preview_jpeg(print_jpeg)
        (folder / files["digital"]).write_bytes(digital_jpeg)
        (folder / files["print"]).write_bytes(print_jpeg)
        (folder / files["preview_digital"]).write_bytes(preview_digital)
        (folder / files["preview_print"]).write_bytes(preview_print)
        if primary_alias:
            (folder / "digital.jpg").write_bytes(digital_jpeg)
            (folder / "print.jpg").write_bytes(print_jpeg)
            (folder / "preview_digital.jpg").write_bytes(preview_digital)
            (folder / "preview_print.jpg").write_bytes(preview_print)
        meta = load_meta(result_id)
        if not meta:
            return False
        _upsert_variant(meta, _variant_entry(variant_id))
        if primary_alias:
            meta["digital"] = "digital.jpg"
            meta["print"] = "print.jpg"
            meta["digital_bytes"] = len(digital_jpeg)
            meta["print_bytes"] = len(print_jpeg)
            meta["preview_only"] = False
            if crop is not None:
                meta["crop"] = crop
            if compliance is not None:
                meta["compliance"] = compliance
            if print_sheet is not None:
                meta["print_sheet"] = print_sheet
            if edit is not None:
                meta["edit_full"] = edit
        if mark_ready:
            meta["full"] = {"status": "ready"}
        _write_meta(result_id, meta)
        log.info(
            "Attached variant jpegs id=%s variant=%s primary=%s ready=%s",
            result_id,
            variant_id,
            primary_alias,
            mark_ready,
        )
        return True
    except Exception as e:
        log.warning(
            "Failed to attach variant jpegs id=%s variant=%s: %s",
            result_id,
            variant_id,
            e,
        )
        return False


def attach_full_jpegs(
    result_id: str,
    digital_jpeg: bytes,
    print_jpeg: bytes,
    *,
    crop: dict[str, Any] | None = None,
    compliance: dict[str, Any] | None = None,
    print_sheet: dict[str, Any] | None = None,
    edit: dict[str, Any] | None = None,
    mark_ready: bool = True,
) -> bool:
    """Write the paid Riverflow full frame (primary alias) + variant files."""
    return attach_variant_jpegs(
        result_id,
        VARIANT_RIVERFLOW,
        digital_jpeg,
        print_jpeg,
        crop=crop,
        compliance=compliance,
        print_sheet=print_sheet,
        edit=edit,
        primary_alias=True,
        mark_ready=mark_ready,
    )


_PAYMENT_META_KEYS = frozenset(
    {
        "paid",
        "paid_at",
        "payment_id",
        "tochka_operation_id",
        "paid_resume",
        "paid_resume_at",
        "resume_payment_id",
        "resume_tochka_operation_id",
    }
)


def clone_result(source_id: str) -> str | None:
    """
    Copy result files into a new unpaid result_id (for same-day process cache hits).
    """
    if not config.RESULTS_ENABLED:
        return None
    if not is_valid_result_id(source_id):
        return None
    src = result_dir(source_id)
    if not (src / "meta.json").is_file():
        return None
    meta = load_meta(source_id)
    if not meta:
        return None
    new_id = new_result_id()
    dst = result_dir(new_id)
    try:
        dst.mkdir(parents=True, exist_ok=False)
        for name in ALLOWED_FILES:
            src_file = src / name
            if src_file.is_file():
                shutil.copy2(src_file, dst / name)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        new_meta = {k: v for k, v in meta.items() if k not in _PAYMENT_META_KEYS}
        new_meta.update(
            {
                "result_id": new_id,
                "saved_at": ts,
                "paid": False,
                "paid_resume": False,
                "cloned_from": source_id,
            }
        )
        (dst / "meta.json").write_text(
            json.dumps(new_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("Cloned result source=%s new=%s", source_id, new_id)
        return new_id
    except Exception as e:
        log.warning("Failed to clone result source=%s: %s", source_id, e)
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
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
        log.warning(
            "payment event=meta_paid_missing_result result_id=%s payment_id=%s",
            result_id,
            payment_id,
        )
        return False
    was_paid = bool(meta.get("paid"))
    ts = paid_at or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta["paid"] = True
    meta["paid_at"] = ts
    meta["payment_id"] = payment_id
    meta["tochka_operation_id"] = tochka_operation_id
    try:
        _write_meta(result_id, meta)
        log.info(
            "payment event=meta_paid result_id=%s payment_id=%s operation_id=%s "
            "paid_at=%s was_paid=%s",
            result_id,
            payment_id,
            tochka_operation_id,
            ts,
            was_paid,
        )
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
    was_paid = bool(meta.get("paid_resume"))
    ts = paid_at or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta["paid_resume"] = True
    meta["paid_resume_at"] = ts
    meta["resume_payment_id"] = payment_id
    meta["resume_tochka_operation_id"] = tochka_operation_id
    try:
        _write_meta(result_id, meta)
        log.info(
            "payment event=meta_paid_resume result_id=%s payment_id=%s "
            "operation_id=%s paid_at=%s was_paid=%s",
            result_id,
            payment_id,
            tochka_operation_id,
            ts,
            was_paid,
        )
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
    if name == "view_digital.jpg":
        rebuilt = ensure_view(result_id)
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


def ensure_view(result_id: str) -> bytes | None:
    """Rebuild full-size watermarked view JPEG from digital.jpg."""
    folder = result_dir(result_id)
    source = folder / "digital.jpg"
    if not source.is_file():
        return None
    try:
        view = make_view_jpeg(source.read_bytes())
        (folder / "view_digital.jpg").write_bytes(view)
        return view
    except Exception as e:
        log.warning("Failed to build view jpeg id=%s: %s", result_id, e)
        return None
