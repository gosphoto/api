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


def load_payment(payment_id: str) -> dict[str, Any] | None:
    if not payment_id:
        return None
    path = _payment_path(payment_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("Failed to read payment %s: %s", payment_id, e)
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
        results.set_paid_resume(
            result_id,
            payment_id=payment_id,
            tochka_operation_id=tochka_operation_id,
            paid_at=paid_at,
        )
    else:
        results.set_paid(
            result_id,
            payment_id=payment_id,
            tochka_operation_id=tochka_operation_id,
            paid_at=paid_at,
        )


def mark_paid(
    payment_id: str,
    *,
    tochka_operation_id: str,
    paid_at: str | None = None,
) -> dict[str, Any] | None:
    record = load_payment(payment_id)
    if not record:
        return None
    if record.get("status") == "paid":
        return record
    ts = paid_at or _now_iso()
    record["status"] = "paid"
    record["paid_at"] = ts
    record["tochka_operation_id"] = tochka_operation_id
    _write_payment(record)
    _apply_unlock(record, tochka_operation_id=tochka_operation_id, paid_at=ts)
    log.info(
        "Payment marked paid payment_id=%s result_id=%s product=%s operation=%s",
        payment_id,
        record.get("result_id"),
        record.get("product") or PRODUCT_PASSPORT,
        tochka_operation_id,
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
    except TochkaError:
        raise
    except Exception as e:
        raise TochkaError(str(e)) from e

    ts = _now_iso()
    record = {
        "payment_id": payment_id,
        "result_id": result_id,
        "product": PRODUCT_PASSPORT,
        "amount_kopecks": config.PRICE_KOPECKS,
        "status": "pending",
        "tochka_operation_id": tochka.operation_id,
        "created_at": ts,
        "paid_at": None,
    }
    _write_payment(record)
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
    """Create Tochka payment for resume-suit unlock (500 ₽)."""
    if not results.is_valid_result_id(result_id):
        raise ValueError("invalid result id")
    meta = results.load_meta(result_id)
    if not meta:
        raise FileNotFoundError("result not found")
    if not meta.get("resume_offer"):
        raise ValueError("resume offer not available")

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
    except TochkaError:
        raise
    except Exception as e:
        raise TochkaError(str(e)) from e

    ts = _now_iso()
    record = {
        "payment_id": payment_id,
        "result_id": result_id,
        "product": PRODUCT_RESUME,
        "amount_kopecks": config.RESUME_PRICE_KOPECKS,
        "status": "pending",
        "tochka_operation_id": tochka.operation_id,
        "created_at": ts,
        "paid_at": None,
    }
    _write_payment(record)
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
        log.warning(
            "Tochka webhook parse failed (invalid JWT/JSON) bytes=%s",
            len(raw_body or ""),
        )
        return {"ok": True, "ignored": True, "reason": "invalid_webhook"}

    log.info(
        "Tochka webhook parsed type=%s status=%s operationId=%s paymentLinkId=%s",
        event.webhook_type,
        event.status,
        event.operation_id,
        event.payment_link_id,
    )

    if event.webhook_type and event.webhook_type != "acquiringInternetPayment":
        log.info("Tochka webhook ignored type=%s", event.webhook_type)
        return {"ok": True, "ignored": True, "reason": "webhook_type"}

    status = (event.status or "").upper()
    if status not in PAID_STATUSES:
        log.info(
            "Tochka webhook ignored status=%s operationId=%s",
            status,
            event.operation_id,
        )
        return {"ok": True, "ignored": True, "reason": "status"}

    record = None
    if event.payment_link_id:
        record = load_payment(event.payment_link_id)
    if record is None and event.operation_id:
        record = find_by_tochka_id(event.operation_id)
    if record is None:
        log.warning(
            "Tochka webhook payment not found paymentLinkId=%s operationId=%s",
            event.payment_link_id,
            event.operation_id,
        )
        return {"ok": True, "ignored": True, "reason": "not_found"}

    if record.get("status") == "paid":
        log.info(
            "Tochka webhook already paid payment_id=%s result_id=%s",
            record.get("payment_id"),
            record.get("result_id"),
        )
        return {"ok": True, "paid": True, "already": True}

    tochka_id = event.operation_id or record.get("tochka_operation_id")
    if not tochka_id:
        log.warning(
            "Tochka webhook missing operationId payment_id=%s",
            record.get("payment_id"),
        )
        return {"ok": True, "ignored": True, "reason": "no_operation_id"}

    mark_paid(record["payment_id"], tochka_operation_id=tochka_id)
    log.info(
        "Tochka webhook activated payment_id=%s result_id=%s product=%s operationId=%s",
        record["payment_id"],
        record["result_id"],
        record.get("product") or PRODUCT_PASSPORT,
        tochka_id,
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
        return False
    if remote.status.upper() not in PAID_STATUSES:
        return False
    mark_paid(
        record["payment_id"],
        tochka_operation_id=remote.operation_id or operation_id,
        paid_at=remote.paid_at,
    )
    return True


def sync_pending_for_result(result_id: str) -> bool:
    """Poll Tochka for pending payments of this result. Returns True if any activated."""
    activated = False
    for record in find_pending_by_result(result_id):
        if activate_if_tochka_paid(record):
            activated = True
    return activated or results.is_paid(result_id) or results.is_paid_resume(result_id)


def sync_all_pending() -> int:
    activated = 0
    for record in find_all_pending():
        if activate_if_tochka_paid(record):
            activated += 1
    if activated:
        log.info("Tochka syncAll activated %s payment(s)", activated)
    return activated
