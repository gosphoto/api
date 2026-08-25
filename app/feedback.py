from __future__ import annotations

import re
import smtplib
import time
from email.message import EmailMessage
from email.utils import formataddr

from . import config

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ALLOWED_PHOTO = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
_RATE: dict[str, list[float]] = {}


class FeedbackValidationError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class FeedbackRateLimitError(Exception):
    detail = "Too many feedback requests; try later"


class FeedbackConfigError(Exception):
    detail = "SMTP is not configured"


class FeedbackSmtpError(Exception):
    def __init__(self, detail: str = "Failed to send email"):
        self.detail = detail
        super().__init__(detail)


def validate_email(value: str) -> str:
    email = (value or "").strip()
    if not email or len(email) > 254 or not _EMAIL_RE.match(email):
        raise FeedbackValidationError(400, "Invalid email")
    return email


def validate_message(value: str) -> str:
    msg = (value or "").strip()
    if len(msg) < config.FEEDBACK_MIN_MESSAGE_CHARS:
        raise FeedbackValidationError(400, "Message is too short")
    if len(msg) > config.FEEDBACK_MAX_MESSAGE_CHARS:
        raise FeedbackValidationError(400, "Message is too long")
    return msg


def validate_full_name(value: str) -> str:
    """Имя Отчество Ф. as on the payment card — for refunds."""
    name = re.sub(r"\s+", " ", (value or "").strip())
    hint = "Укажите Имя Отчество Ф., например: Иван Сергеевич П."
    if len(name) < config.FEEDBACK_MIN_FULL_NAME_CHARS:
        raise FeedbackValidationError(400, hint)
    if len(name) > config.FEEDBACK_MAX_FULL_NAME_CHARS:
        raise FeedbackValidationError(400, "ФИО слишком длинное")
    parts = [p for p in name.split(" ") if p]
    if len(parts) < 3:
        raise FeedbackValidationError(400, hint)
    return name


def validate_photo(
    filename: str | None, content_type: str | None, data: bytes | None
) -> tuple[str, bytes] | None:
    if data is None or data == b"":
        return None
    if len(data) > config.FEEDBACK_MAX_PHOTO_BYTES:
        raise FeedbackValidationError(413, "Photo too large")
    ct = (content_type or "").lower().strip()
    name = (filename or "photo").lower()
    if ct not in _ALLOWED_PHOTO:
        if name.endswith((".jpg", ".jpeg")):
            ct = "image/jpeg"
        elif name.endswith(".png"):
            ct = "image/png"
        elif name.endswith(".webp"):
            ct = "image/webp"
        else:
            raise FeedbackValidationError(400, "Photo must be JPEG, PNG, or WebP")
    ext = _ALLOWED_PHOTO["image/jpeg" if ct == "image/jpg" else ct]
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", (filename or f"photo{ext}"))[:80]
    if "." not in safe:
        safe = f"{safe}{ext}"
    return safe, data


def check_rate_limit(ip: str) -> None:
    now = time.time()
    window = float(config.FEEDBACK_RATE_WINDOW_SEC)
    limit = int(config.FEEDBACK_RATE_LIMIT)
    key = (ip or "unknown").strip() or "unknown"
    hits = [t for t in _RATE.get(key, []) if now - t < window]
    if len(hits) >= limit:
        _RATE[key] = hits
        raise FeedbackRateLimitError()
    hits.append(now)
    _RATE[key] = hits


def build_feedback_email(
    *,
    email: str,
    full_name: str,
    message: str,
    client_ip: str,
    user_agent: str,
    photo: tuple[str, bytes] | None,
) -> EmailMessage:
    snippet = message.replace("\n", " ").strip()
    if len(snippet) > 60:
        snippet = snippet[:57] + "..."
    msg = EmailMessage()
    msg["Subject"] = f"[GoSphoto feedback] {snippet}"
    msg["From"] = formataddr(("GoSphoto feedback", config.SMTP_FROM))
    msg["To"] = config.FEEDBACK_TO
    msg["Reply-To"] = email
    msg.set_content(
        "\n".join(
            [
                f"From: {email}",
                f"ФИО: {full_name}",
                f"IP: {client_ip}",
                f"User-Agent: {user_agent}",
                "",
                message,
            ]
        )
    )
    if photo:
        filename, data = photo
        subtype = "jpeg"
        lower = filename.lower()
        if lower.endswith(".png"):
            subtype = "png"
        elif lower.endswith(".webp"):
            subtype = "webp"
        msg.add_attachment(
            data, maintype="image", subtype=subtype, filename=filename
        )
    return msg


def send_feedback_email(msg: EmailMessage) -> None:
    if not config.SMTP_PASSWORD or not config.SMTP_HOST or not config.SMTP_USER:
        raise FeedbackConfigError()
    try:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
            smtp.send_message(msg)
    except FeedbackConfigError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface as 502
        raise FeedbackSmtpError(str(exc) or "Failed to send email") from exc
