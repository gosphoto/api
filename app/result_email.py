"""Send paid result JPEGs to a customer email via SMTP."""

from __future__ import annotations

import time
from email.message import EmailMessage
from email.utils import formataddr

from . import config
from .feedback import (
    FeedbackConfigError,
    FeedbackRateLimitError,
    FeedbackSmtpError,
    FeedbackValidationError,
    send_feedback_email,
    validate_email,
)

_RATE: dict[str, list[float]] = {}


class ResultEmailRateLimitError(FeedbackRateLimitError):
    detail = "Слишком много запросов на отправку; попробуйте позже"


def check_rate_limit(ip: str) -> None:
    now = time.time()
    window = float(config.RESULT_EMAIL_RATE_WINDOW_SEC)
    limit = int(config.RESULT_EMAIL_RATE_LIMIT)
    key = (ip or "unknown").strip() or "unknown"
    hits = [t for t in _RATE.get(key, []) if now - t < window]
    if len(hits) >= limit:
        _RATE[key] = hits
        raise ResultEmailRateLimitError()
    hits.append(now)
    _RATE[key] = hits


def build_result_email(
    *,
    email: str,
    result_id: str,
    digital_jpeg: bytes,
    print_jpeg: bytes,
) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = "Ваше фото на документы — Госфото"
    msg["From"] = formataddr(("Госфото", config.SMTP_FROM))
    msg["To"] = email
    msg.set_content(
        "\n".join(
            [
                "Здравствуйте!",
                "",
                "Во вложении готовые файлы после оплаты:",
                "• gosphoto-passport.jpg — JPEG 35×45 для Госуслуг",
                "• gosphoto-10x15.jpg — лист 10×15 см (4 фото) для печати",
                "",
                f"Страница результата: {config.PUBLIC_BASE_URL}/result/{result_id}",
                "",
                "Если письмо пришло по ошибке — просто удалите его.",
                "",
                "— Госфото",
                "https://gosphoto.ru",
            ]
        )
    )
    msg.add_attachment(
        digital_jpeg,
        maintype="image",
        subtype="jpeg",
        filename="gosphoto-passport.jpg",
    )
    msg.add_attachment(
        print_jpeg,
        maintype="image",
        subtype="jpeg",
        filename="gosphoto-10x15.jpg",
    )
    return msg


def send_result_email(msg: EmailMessage) -> None:
    send_feedback_email(msg)


__all__ = [
    "FeedbackConfigError",
    "FeedbackRateLimitError",
    "FeedbackSmtpError",
    "FeedbackValidationError",
    "ResultEmailRateLimitError",
    "build_result_email",
    "check_rate_limit",
    "send_result_email",
    "validate_email",
]
