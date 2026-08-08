"""Tochka Bank acquiring client (ported from kkalscan HttpTochkaClient)."""

from __future__ import annotations

import base64
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.exceptions import InvalidTokenError

from . import config

log = logging.getLogger("gosphoto-gate")

# https://enter.tochka.com/doc/openapi/static/keys/public
_TOCHKA_MODULUS_B64URL = (
    "rwm77av7GIttq-JF1itEgLCGEZW_zz16RlUQVYlLbJtyRSu61fCec_rroP6PxjXU2uLzUOaGaLgA"
    "PeUZAJrGuVp9nryKgbZceHckdHDYgJd9TsdJ1MYUsXaOb9joN9vmsCscBx1lwSlFQyNQsHUsrjuD"
    "k-opf6RCuazRQ9gkoDCX70HV8WBMFoVm-YWQKJHZEaIQxg_DU4gMFyKRkDGKsYKA0POL-UgWA1qk"
    "g6nHY5BOMKaqxbc5ky87muWB5nNk4mfmsckyFv9j1gBiXLKekA_y4UwG2o1pbOLpJS3bP_c95rm4"
    "M9ZBmGXqfOQhbjz8z-s9C11i-jmOQ2ByohS-ST3E5sqBzIsxxrxyQDTw--bZNhzpbciyYW4Gfkkq"
    "yeYoOPd_84jPTBDKQXssvj8ZOj2XboS77tvEO1n1WlwUzh8HPCJod5_fEgSXuozpJtOggXBv0C2p"
    "s7yXlDZf-7Jar0UYc_NJEHJF-xShlqd6Q3sVL02PhSCM-ibn9DN9BKmD"
)
_TOCHKA_EXPONENT_B64URL = "AQAB"

PAID_STATUSES = frozenset({"PAID", "SUCCESS", "APPROVED"})


@dataclass
class TochkaPayment:
    operation_id: str
    payment_url: str


@dataclass
class TochkaPaymentStatus:
    operation_id: str
    payment_link_id: str | None
    status: str
    paid_at: str | None = None


@dataclass
class TochkaWebhookEvent:
    operation_id: str | None
    payment_link_id: str | None
    status: str
    webhook_type: str | None = None


class TochkaError(Exception):
    pass


def _b64url_to_int(value: str) -> int:
    pad = "=" * (-len(value) % 4)
    return int.from_bytes(base64.urlsafe_b64decode(value + pad), "big")


def _tochka_public_key():
    n = _b64url_to_int(_TOCHKA_MODULUS_B64URL)
    e = _b64url_to_int(_TOCHKA_EXPONENT_B64URL)
    return rsa.RSAPublicNumbers(e, n).public_key()


def parse_webhook_jwt(raw_body: str) -> TochkaWebhookEvent | None:
    trimmed = raw_body.strip().strip('"')
    if not trimmed.startswith("eyJ"):
        return None
    try:
        payload = jwt.decode(
            trimmed,
            _tochka_public_key(),
            algorithms=["RS256"],
            options={
                "verify_aud": False,
                "verify_iss": False,
                "verify_exp": False,
                "verify_nbf": False,
            },
            leeway=60,
        )
    except InvalidTokenError as e:
        log.warning("Tochka webhook JWT verify failed: %s", e)
        return None
    if not isinstance(payload, dict):
        return None
    return TochkaWebhookEvent(
        operation_id=_str_or_none(payload.get("operationId")),
        payment_link_id=_str_or_none(payload.get("paymentLinkId")),
        status=_str_or_none(payload.get("status")) or "unknown",
        webhook_type=_str_or_none(payload.get("webhookType")),
    )


def parse_webhook_json(raw_body: str) -> TochkaWebhookEvent | None:
    text = raw_body.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return TochkaWebhookEvent(
        operation_id=_str_or_none(
            data.get("payment_id") or data.get("operationId") or data.get("operation_id")
        ),
        payment_link_id=_str_or_none(
            data.get("payment_link_id")
            or data.get("paymentLinkId")
            or data.get("payment_linkId")
        ),
        status=_str_or_none(data.get("status")) or "unknown",
        webhook_type=_str_or_none(data.get("webhookType") or data.get("webhook_type")),
    )


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


class HttpTochkaClient:
    def __init__(
        self,
        *,
        access_token: str | None = None,
        merchant_id: str | None = None,
        customer_code: str | None = None,
        api_base_url: str | None = None,
        timeout: float = 30.0,
    ):
        self.access_token = (access_token if access_token is not None else config.TOCHKA_ACCESS_TOKEN).strip()
        self.merchant_id = (merchant_id if merchant_id is not None else config.TOCHKA_MERCHANT_ID).strip()
        self._customer_code_override = (
            customer_code if customer_code is not None else config.TOCHKA_CUSTOMER_CODE
        ).strip()
        self.api_base_url = (
            api_base_url if api_base_url is not None else config.TOCHKA_API_BASE_URL
        ).rstrip("/")
        self.timeout = timeout
        self._cached_customer_code: str | None = None
        self._lock = threading.Lock()

    def create_payment(
        self,
        amount_kopecks: int,
        description: str,
        metadata: dict[str, str],
    ) -> TochkaPayment:
        payment_link_id = metadata.get("payment_link_id")
        if not payment_link_id:
            raise TochkaError("payment_link_id обязателен")
        customer_code = self.resolve_customer_code()
        amount_rub = f"{amount_kopecks / 100.0:.2f}"
        data: dict[str, Any] = {
            "customerCode": customer_code,
            "amount": amount_rub,
            "purpose": description,
            "paymentMode": ["card", "sbp"],
            "paymentLinkId": payment_link_id,
        }
        if self.merchant_id:
            data["merchantId"] = self.merchant_id
        if metadata.get("redirect_url"):
            data["redirectUrl"] = metadata["redirect_url"]
        if metadata.get("fail_redirect_url"):
            data["failRedirectUrl"] = metadata["fail_redirect_url"]

        payload = {"Data": data}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.api_base_url}/uapi/acquiring/v1.0/payments",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                content=json.dumps(payload),
            )
        body = response.text
        if response.status_code >= 400:
            log.warning(
                "Tochka create payment failed: HTTP %s %s",
                response.status_code,
                body[:500],
            )
            raise TochkaError("Не удалось создать платёж в Точке")

        root = response.json()
        data_obj = root.get("Data") if isinstance(root, dict) else None
        if not isinstance(data_obj, dict):
            raise TochkaError("Некорректный ответ Точки")
        operation_id = _str_or_none(data_obj.get("operationId"))
        payment_url = _str_or_none(data_obj.get("paymentLink") or data_obj.get("paymentUrl"))
        if not operation_id or not payment_url:
            raise TochkaError("Точка не вернула operationId/paymentLink")
        log.info(
            "Tochka payment created: operationId=%s paymentLinkId=%s",
            operation_id,
            payment_link_id,
        )
        return TochkaPayment(operation_id=operation_id, payment_url=payment_url)

    def get_payment_status(self, operation_id: str) -> TochkaPaymentStatus | None:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.api_base_url}/uapi/acquiring/v1.0/payments/{operation_id}",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Accept": "application/json",
                },
            )
        if response.status_code >= 400:
            log.warning(
                "Tochka get payment failed: HTTP %s %s",
                response.status_code,
                response.text[:500],
            )
            return None
        root = response.json()
        data_obj = root.get("Data") if isinstance(root, dict) else None
        if not isinstance(data_obj, dict):
            return None
        operation = data_obj.get("Operation")
        if isinstance(operation, list):
            operation = operation[0] if operation else None
        if not isinstance(operation, dict):
            return None
        status = _str_or_none(operation.get("status"))
        if not status:
            return None
        return TochkaPaymentStatus(
            operation_id=_str_or_none(operation.get("operationId")) or operation_id,
            payment_link_id=_str_or_none(operation.get("paymentLinkId")),
            status=status,
            paid_at=_str_or_none(operation.get("paidAt")),
        )

    def parse_webhook(self, raw_body: str, signature: str | None = None) -> TochkaWebhookEvent | None:
        _ = signature
        return parse_webhook_jwt(raw_body)

    def resolve_customer_code(self) -> str:
        if self._customer_code_override:
            return self._customer_code_override
        with self._lock:
            if self._cached_customer_code:
                return self._cached_customer_code
            code = self._fetch_business_customer_code()
            self._cached_customer_code = code
            log.info("Tochka customerCode resolved: %s", code)
            return code

    def _fetch_business_customer_code(self) -> str:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.api_base_url}/uapi/open-banking/v1.0/customers",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Accept": "application/json",
                },
            )
        if response.status_code >= 400:
            log.warning(
                "Tochka customers list failed: HTTP %s %s",
                response.status_code,
                response.text[:500],
            )
            raise TochkaError("Не удалось получить customerCode из Точки")
        root = response.json()
        data = root.get("Data") if isinstance(root, dict) else None
        if not isinstance(data, dict):
            raise TochkaError("Точка не вернула список клиентов")
        direct = _str_or_none(data.get("customerCode"))
        if direct:
            return direct
        customers_raw = data.get("Customer")
        customers: list[dict] = []
        if isinstance(customers_raw, dict):
            customers = [customers_raw]
        elif isinstance(customers_raw, list):
            customers = [c for c in customers_raw if isinstance(c, dict)]
        if not customers:
            raise TochkaError("В Точке не найден customerCode")
        for customer in customers:
            if (_str_or_none(customer.get("customerType")) or "").lower() == "business":
                code = _str_or_none(customer.get("customerCode"))
                if code:
                    return code
        if len(customers) == 1:
            code = _str_or_none(customers[0].get("customerCode"))
            if code:
                return code
        raise TochkaError("Несколько клиентов в Точке — задайте TOCHKA_CUSTOMER_CODE вручную")


class StubTochkaClient:
    """Local/CI stub — no network. Webhook accepts plain JSON."""

    def __init__(self) -> None:
        self._statuses: dict[str, TochkaPaymentStatus] = {}

    def create_payment(
        self,
        amount_kopecks: int,
        description: str,
        metadata: dict[str, str],
    ) -> TochkaPayment:
        _ = amount_kopecks, description
        link = metadata.get("payment_link_id") or "x"
        operation_id = f"tochka_{link[:8]}"
        self._statuses[operation_id] = TochkaPaymentStatus(
            operation_id=operation_id,
            payment_link_id=metadata.get("payment_link_id"),
            status="CREATED",
            paid_at=None,
        )
        return TochkaPayment(
            operation_id=operation_id,
            payment_url=f"https://pay.tochka.example/{operation_id}",
        )

    def get_payment_status(self, operation_id: str) -> TochkaPaymentStatus | None:
        return self._statuses.get(operation_id)

    def mark_approved(self, operation_id: str, paid_at: str | None = None) -> None:
        current = self._statuses.get(operation_id)
        if not current:
            return
        self._statuses[operation_id] = TochkaPaymentStatus(
            operation_id=current.operation_id,
            payment_link_id=current.payment_link_id,
            status="APPROVED",
            paid_at=paid_at or current.paid_at,
        )

    def parse_webhook(self, raw_body: str, signature: str | None = None) -> TochkaWebhookEvent | None:
        if signature is not None and signature != "test-signature":
            return None
        return parse_webhook_json(raw_body)


_client: HttpTochkaClient | StubTochkaClient | None = None
_client_lock = threading.Lock()


def get_tochka_client() -> HttpTochkaClient | StubTochkaClient:
    global _client
    with _client_lock:
        if _client is None:
            if config.TOCHKA_ACCESS_TOKEN:
                _client = HttpTochkaClient()
            else:
                log.warning("TOCHKA_ACCESS_TOKEN unset — using StubTochkaClient")
                _client = StubTochkaClient()
        return _client


def reset_tochka_client() -> None:
    """Test helper."""
    global _client
    with _client_lock:
        _client = None
