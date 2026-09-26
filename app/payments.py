"""File-backed payments tied to result_id (Tochka paymentLinkId = payment_id)."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config
from . import results
from .tochka import (
    PAID_STATUSES,
    TochkaError,
    get_tochka_client,
)

log = logging.getLogger("gosphoto-gate")

PRODUCT_PASSPORT = "passport"
PRODUCT_RESUME = "resume"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _pay_log(event: str, level: int = logging.INFO, **fields: Any) -> None:
    """Structured payment trail — grep: `payment event=`."""
    parts = [f"payment event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    log.log(level, " ".join(parts))


def _payments_dir() -> Path:
    path = Path(config.PAYMENTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _payment_path(payment_id: str) -> Path:
    return _payments_dir() / f"{payment_id}.json"


def _write_payment(record: dict[str, Any]) -> None:
    path = _payment_path(record["payment_id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
    _pay_log(
        "file_written",
        payment_id=record.get("payment_id"),
        result_id=record.get("result_id"),
        product=record.get("product") or PRODUCT_PASSPORT,
        status=record.get("status"),
        amount_kopecks=record.get("amount_kopecks"),
        path=str(path),
    )


def load_payment(payment_id: str) -> dict[str, Any] | None:
    if not payment_id:
        return None
    path = _payment_path(payment_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _pay_log(
            "file_read_failed",
            logging.WARNING,
            payment_id=payment_id,
            path=str(path),
            error=e,
        )
        return None
    return data if isinstance(data, dict) else None


def find_by_tochka_id(operation_id: str) -> dict[str, Any] | None:
    if not operation_id:
        return None
    for path in _payments_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("tochka_operation_id") == operation_id:
            return data
    return None


def find_pending_by_result(result_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _payments_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if (
            isinstance(data, dict)
            and data.get("result_id") == result_id
            and data.get("status") == "pending"
        ):
            out.append(data)
    return out


def _product_of(record: dict[str, Any]) -> str:
    return str(record.get("product") or PRODUCT_PASSPORT)


def latest_pending(result_id: str, product: str) -> dict[str, Any] | None:
    matches = [
        p for p in find_pending_by_result(result_id) if _product_of(p) == product
    ]
    if not matches:
        return None
    return max(matches, key=lambda p: str(p.get("created_at") or ""))


def _reuse_pending_checkout(
    result_id: str, product: str
) -> dict[str, Any] | None:
    """Return existing Tochka link instead of a second charge."""
    record = latest_pending(result_id, product)
    if not record:
        return None
    payment_url = record.get("payment_url")
    if not payment_url:
        return None
    is_resume = product == PRODUCT_RESUME
    price = resume_price_rub() if is_resume else price_rub()
    kopecks = (
        config.RESUME_PRICE_KOPECKS if is_resume else config.PRICE_KOPECKS
    )
    body: dict[str, Any] = {
        "ok": True,
        "paid": False,
        "payment_required": True,
        "product": product,
        "payment_id": record["payment_id"],
        "payment_url": payment_url,
        "result_id": result_id,
        "price_kopecks": kopecks,
        "price_rub": price,
        "reused": True,
    }
    if is_resume:
        body["paid_resume"] = False
    _pay_log(
        "checkout_reused",
        payment_id=record["payment_id"],
        result_id=result_id,
        product=product,
        amount_rub=price,
    )
    return body


def find_all_pending() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in _payments_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("status") == "pending":
            out.append(data)
    return out


def _apply_unlock(record: dict[str, Any], *, tochka_operation_id: str, paid_at: str) -> None:
    product = record.get("product") or PRODUCT_PASSPORT
    result_id = record["result_id"]
    payment_id = record["payment_id"]
    if product == PRODUCT_RESUME:
        ok = results.set_paid_resume(
            result_id,
            payment_id=payment_id,
            tochka_operation_id=tochka_operation_id,
            paid_at=paid_at,
        )
    else:
        ok = results.set_paid(
            result_id,
            payment_id=payment_id,
            tochka_operation_id=tochka_operation_id,
            paid_at=paid_at,
        )
    _pay_log(
        "unlock_applied" if ok else "unlock_failed",
        logging.INFO if ok else logging.ERROR,
        payment_id=payment_id,
        result_id=result_id,
        product=product,
        operation_id=tochka_operation_id,
        paid_at=paid_at,
        meta_ok=ok,
    )


def mark_paid(
    payment_id: str,
    *,
    tochka_operation_id: str,
    paid_at: str | None = None,
    source: str = "unknown",
) -> dict[str, Any] | None:
    record = load_payment(payment_id)
    if not record:
        _pay_log(
            "mark_paid_missing_file",
            logging.ERROR,
            payment_id=payment_id,
            operation_id=tochka_operation_id,
            source=source,
            payments_dir=_payments_dir(),
        )
        return None
    if record.get("status") == "paid":
        _pay_log(
            "mark_paid_already",
            payment_id=payment_id,
            result_id=record.get("result_id"),
            product=record.get("product") or PRODUCT_PASSPORT,
            source=source,
        )
        return record
    ts = paid_at or _now_iso()
    record["status"] = "paid"
    record["paid_at"] = ts
    record["tochka_operation_id"] = tochka_operation_id
    _write_payment(record)
    _apply_unlock(record, tochka_operation_id=tochka_operation_id, paid_at=ts)
    _pay_log(
        "marked_paid",
        payment_id=payment_id,
        result_id=record.get("result_id"),
        product=record.get("product") or PRODUCT_PASSPORT,
        amount_kopecks=record.get("amount_kopecks"),
        operation_id=tochka_operation_id,
        paid_at=ts,
        source=source,
    )
    return record


def price_rub() -> int:
    return max(0, config.PRICE_KOPECKS) // 100


def resume_price_rub() -> int:
    return max(0, config.RESUME_PRICE_KOPECKS) // 100


def create_checkout(result_id: str) -> dict[str, Any]:
    """Create Tochka payment for passport unlock (or free-unlock)."""
    if not results.is_valid_result_id(result_id):
        raise ValueError("invalid result id")
    meta = results.load_meta(result_id)
    if not meta:
        raise FileNotFoundError("result not found")

    sync_pending_for_result(result_id)
    if results.is_paid(result_id) or config.FREE_DOWNLOAD_UNLOCK:
        if not results.is_paid(result_id):
            payment_id = str(uuid.uuid4())
            ts = _now_iso()
            record = {
                "payment_id": payment_id,
                "result_id": result_id,
                "product": PRODUCT_PASSPORT,
                "amount_kopecks": config.PRICE_KOPECKS,
                "status": "free_unlock",
                "tochka_operation_id": f"free_{payment_id}",
                "created_at": ts,
                "paid_at": ts,
            }
            _write_payment(record)
            results.set_paid(
                result_id,
                payment_id=payment_id,
                tochka_operation_id=record["tochka_operation_id"],
                paid_at=ts,
            )
            _pay_log(
                "checkout_free_unlock",
                payment_id=payment_id,
                result_id=result_id,
                product=PRODUCT_PASSPORT,
                amount_rub=price_rub(),
            )
        else:
            _pay_log(
                "checkout_already_paid",
                result_id=result_id,
                product=PRODUCT_PASSPORT,
            )
        return {
            "ok": True,
            "paid": True,
            "payment_required": False,
            "product": PRODUCT_PASSPORT,
            "result_id": result_id,
            "price_kopecks": config.PRICE_KOPECKS,
            "price_rub": price_rub(),
            "message": "Скачивание уже доступно",
        }

    reused = _reuse_pending_checkout(result_id, PRODUCT_PASSPORT)
    if reused:
        return reused

    payment_id = str(uuid.uuid4())
    metadata = {
        "payment_link_id": payment_id,
        "result_id": result_id,
        "product": PRODUCT_PASSPORT,
    }
    base = config.PUBLIC_BASE_URL.strip()
    if base.lower().startswith("https://"):
        metadata["redirect_url"] = f"{base}/result/{result_id}?paid=1"
        metadata["fail_redirect_url"] = f"{base}/result/{result_id}?paid=0"

    client = get_tochka_client()
    try:
        tochka = client.create_payment(
            amount_kopecks=config.PRICE_KOPECKS,
            description=f"Госфото — скачивание фото ({price_rub()} ₽)",
            metadata=metadata,
        )
    except TochkaError as e:
        _pay_log(
            "checkout_tochka_error",
            logging.ERROR,
            result_id=result_id,
            product=PRODUCT_PASSPORT,
            payment_id=payment_id,
            error=e,
        )
        raise
    except Exception as e:
        _pay_log(
            "checkout_tochka_error",
            logging.ERROR,
            result_id=result_id,
            product=PRODUCT_PASSPORT,
            payment_id=payment_id,
            error=e,
        )
        raise TochkaError(str(e)) from e

    ts = _now_iso()
    record = {
        "payment_id": payment_id,
        "result_id": result_id,
        "product": PRODUCT_PASSPORT,
        "amount_kopecks": config.PRICE_KOPECKS,
        "status": "pending",
        "tochka_operation_id": tochka.operation_id,
        "payment_url": tochka.payment_url,
        "created_at": ts,
        "paid_at": None,
    }
    _write_payment(record)
    _pay_log(
        "checkout_created",
        payment_id=payment_id,
        result_id=result_id,
        product=PRODUCT_PASSPORT,
        amount_rub=price_rub(),
        operation_id=tochka.operation_id,
        status="pending",
    )
    return {
        "ok": True,
        "paid": False,
        "payment_required": True,
        "product": PRODUCT_PASSPORT,
        "payment_id": payment_id,
        "payment_url": tochka.payment_url,
        "result_id": result_id,
        "price_kopecks": config.PRICE_KOPECKS,
        "price_rub": price_rub(),
    }


def create_checkout_resume(result_id: str) -> dict[str, Any]:
    """Create Tochka payment for resume-suit unlock (300 ₽)."""
    if not results.is_valid_result_id(result_id):
        raise ValueError("invalid result id")
    meta = results.load_meta(result_id)
    if not meta:
        raise FileNotFoundError("result not found")
    if not meta.get("resume_offer"):
        raise ValueError("resume offer not available")

    sync_pending_for_result(result_id)
    if results.is_paid_resume(result_id) or config.FREE_DOWNLOAD_UNLOCK:
        if not results.is_paid_resume(result_id):
            payment_id = str(uuid.uuid4())
            ts = _now_iso()
            record = {
                "payment_id": payment_id,
                "result_id": result_id,
                "product": PRODUCT_RESUME,
                "amount_kopecks": config.RESUME_PRICE_KOPECKS,
                "status": "free_unlock",
                "tochka_operation_id": f"free_resume_{payment_id}",
                "created_at": ts,
                "paid_at": ts,
            }
            _write_payment(record)
            results.set_paid_resume(
                result_id,
                payment_id=payment_id,
                tochka_operation_id=record["tochka_operation_id"],
                paid_at=ts,
            )
            _pay_log(
                "checkout_free_unlock",
                payment_id=payment_id,
                result_id=result_id,
                product=PRODUCT_RESUME,
                amount_rub=resume_price_rub(),
            )
        else:
            _pay_log(
                "checkout_already_paid",
                result_id=result_id,
                product=PRODUCT_RESUME,
            )
        return {
            "ok": True,
            "paid": True,
            "paid_resume": True,
            "payment_required": False,
            "product": PRODUCT_RESUME,
            "result_id": result_id,
            "price_kopecks": config.RESUME_PRICE_KOPECKS,
            "price_rub": resume_price_rub(),
            "message": "Фото для резюме уже доступно",
        }

    reused = _reuse_pending_checkout(result_id, PRODUCT_RESUME)
    if reused:
        return reused

    payment_id = str(uuid.uuid4())
    metadata = {
        "payment_link_id": payment_id,
        "result_id": result_id,
        "product": PRODUCT_RESUME,
    }
    base = config.PUBLIC_BASE_URL.strip()
    if base.lower().startswith("https://"):
        metadata["redirect_url"] = f"{base}/result/{result_id}?paid_resume=1"
        metadata["fail_redirect_url"] = f"{base}/result/{result_id}?paid_resume=0"

    client = get_tochka_client()
    try:
        tochka = client.create_payment(
            amount_kopecks=config.RESUME_PRICE_KOPECKS,
            description=(
                f"Госфото — фото для резюме ({resume_price_rub()} ₽)"
            ),
            metadata=metadata,
        )
    except TochkaError as e:
        _pay_log(
            "checkout_tochka_error",
            logging.ERROR,
            result_id=result_id,
            product=PRODUCT_RESUME,
            payment_id=payment_id,
            error=e,
        )
        raise
    except Exception as e:
        _pay_log(
            "checkout_tochka_error",
            logging.ERROR,
            result_id=result_id,
            product=PRODUCT_RESUME,
            payment_id=payment_id,
            error=e,
        )
        raise TochkaError(str(e)) from e

    ts = _now_iso()
    record = {
        "payment_id": payment_id,
        "result_id": result_id,
        "product": PRODUCT_RESUME,
        "amount_kopecks": config.RESUME_PRICE_KOPECKS,
        "status": "pending",
        "tochka_operation_id": tochka.operation_id,
        "payment_url": tochka.payment_url,
        "created_at": ts,
        "paid_at": None,
    }
    _write_payment(record)
    _pay_log(
        "checkout_created",
        payment_id=payment_id,
        result_id=result_id,
        product=PRODUCT_RESUME,
        amount_rub=resume_price_rub(),
        operation_id=tochka.operation_id,
        status="pending",
    )
    return {
        "ok": True,
        "paid": False,
        "paid_resume": False,
        "payment_required": True,
        "product": PRODUCT_RESUME,
        "payment_id": payment_id,
        "payment_url": tochka.payment_url,
        "result_id": result_id,
        "price_kopecks": config.RESUME_PRICE_KOPECKS,
        "price_rub": resume_price_rub(),
    }


def handle_webhook(raw_body: str, signature: str | None = None) -> dict[str, Any]:
    client = get_tochka_client()
    # Stub accepts plain JSON; HttpTochkaClient verifies JWT only.
    event = client.parse_webhook(raw_body, signature)
    if event is None:
        _pay_log(
            "webhook_parse_failed",
            logging.WARNING,
            bytes=len(raw_body or ""),
            has_signature=bool(signature),
        )
        return {"ok": True, "ignored": True, "reason": "invalid_webhook"}

    _pay_log(
        "webhook_parsed",
        webhook_type=event.webhook_type,
        status=event.status,
        operation_id=event.operation_id,
        payment_link_id=event.payment_link_id,
    )

    if event.webhook_type and event.webhook_type != "acquiringInternetPayment":
        _pay_log("webhook_ignored", reason="webhook_type", webhook_type=event.webhook_type)
        return {"ok": True, "ignored": True, "reason": "webhook_type"}

    status = (event.status or "").upper()
    if status not in PAID_STATUSES:
        _pay_log(
            "webhook_ignored",
            reason="status",
            status=status,
            operation_id=event.operation_id,
            payment_link_id=event.payment_link_id,
        )
        return {"ok": True, "ignored": True, "reason": "status"}

    record = None
    if event.payment_link_id:
        record = load_payment(event.payment_link_id)
    if record is None and event.operation_id:
        record = find_by_tochka_id(event.operation_id)
    if record is None:
        _pay_log(
            "webhook_not_found",
            logging.ERROR,
            payment_link_id=event.payment_link_id,
            operation_id=event.operation_id,
            payments_dir=_payments_dir(),
        )
        return {"ok": True, "ignored": True, "reason": "not_found"}

    if record.get("status") == "paid":
        _pay_log(
            "webhook_already_paid",
            payment_id=record.get("payment_id"),
            result_id=record.get("result_id"),
            product=record.get("product") or PRODUCT_PASSPORT,
        )
        return {"ok": True, "paid": True, "already": True}

    tochka_id = event.operation_id or record.get("tochka_operation_id")
    if not tochka_id:
        _pay_log(
            "webhook_no_operation_id",
            logging.WARNING,
            payment_id=record.get("payment_id"),
        )
        return {"ok": True, "ignored": True, "reason": "no_operation_id"}

    mark_paid(
        record["payment_id"],
        tochka_operation_id=tochka_id,
        source="webhook",
    )
    _pay_log(
        "webhook_activated",
        payment_id=record["payment_id"],
        result_id=record["result_id"],
        product=record.get("product") or PRODUCT_PASSPORT,
        operation_id=tochka_id,
        amount_kopecks=record.get("amount_kopecks"),
    )
    return {
        "ok": True,
        "paid": True,
        "payment_id": record["payment_id"],
        "result_id": record["result_id"],
        "product": record.get("product") or PRODUCT_PASSPORT,
    }


def activate_if_tochka_paid(record: dict[str, Any]) -> bool:
    if record.get("status") == "paid":
        return False
    operation_id = record.get("tochka_operation_id")
    if not operation_id:
        return False
    remote = get_tochka_client().get_payment_status(operation_id)
    if not remote:
        _pay_log(
            "sync_poll_no_remote",
            logging.WARNING,
            payment_id=record.get("payment_id"),
            result_id=record.get("result_id"),
            operation_id=operation_id,
        )
        return False
    if remote.status.upper() not in PAID_STATUSES:
        return False
    mark_paid(
        record["payment_id"],
        tochka_operation_id=remote.operation_id or operation_id,
        paid_at=remote.paid_at,
        source="sync_poll",
    )
    _pay_log(
        "sync_poll_activated",
        payment_id=record.get("payment_id"),
        result_id=record.get("result_id"),
        product=record.get("product") or PRODUCT_PASSPORT,
        operation_id=remote.operation_id or operation_id,
        amount_kopecks=record.get("amount_kopecks"),
    )
    return True


def sync_pending_for_result(result_id: str) -> bool:
    """Poll Tochka for pending payments of this result. Returns True if any activated."""
    pending = find_pending_by_result(result_id)
    if pending:
        _pay_log(
            "sync_result_start",
            result_id=result_id,
            pending_count=len(pending),
        )
    activated = False
    for record in pending:
        if activate_if_tochka_paid(record):
            activated = True
    paid_now = results.is_paid(result_id) or results.is_paid_resume(result_id)
    if activated:
        _pay_log("sync_result_done", result_id=result_id, activated=True, paid=paid_now)
    return activated or paid_now


def sync_all_pending() -> int:
    pending = find_all_pending()
    activated = 0
    for record in pending:
        if activate_if_tochka_paid(record):
            activated += 1
    if pending:
        _pay_log(
            "sync_all_done",
            pending_count=len(pending),
            activated=activated,
            payments_dir=_payments_dir(),
        )
    return activated
