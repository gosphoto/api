import pytest

from app import config
from app import feedback


def test_validate_email_ok():
    assert feedback.validate_email("  User@Example.com ") == "User@Example.com"


def test_validate_email_rejects_bad():
    with pytest.raises(feedback.FeedbackValidationError) as e:
        feedback.validate_email("not-an-email")
    assert e.value.status_code == 400


def test_validate_message_bounds(monkeypatch):
    monkeypatch.setattr(config, "FEEDBACK_MIN_MESSAGE_CHARS", 10)
    monkeypatch.setattr(config, "FEEDBACK_MAX_MESSAGE_CHARS", 20)
    assert feedback.validate_message("  hello world  ") == "hello world"
    with pytest.raises(feedback.FeedbackValidationError):
        feedback.validate_message("short")
    with pytest.raises(feedback.FeedbackValidationError):
        feedback.validate_message("x" * 21)


def test_validate_photo_optional_none():
    assert feedback.validate_photo(None, None, None) is None


def test_validate_photo_rejects_type_and_size(monkeypatch):
    monkeypatch.setattr(config, "FEEDBACK_MAX_PHOTO_BYTES", 10)
    with pytest.raises(feedback.FeedbackValidationError) as e:
        feedback.validate_photo("a.gif", "image/gif", b"123")
    assert e.value.status_code == 400
    with pytest.raises(feedback.FeedbackValidationError) as e:
        feedback.validate_photo("a.jpg", "image/jpeg", b"0123456789ABC")
    assert e.value.status_code == 413


def test_rate_limit_trips(monkeypatch):
    monkeypatch.setattr(config, "FEEDBACK_RATE_LIMIT", 2)
    monkeypatch.setattr(config, "FEEDBACK_RATE_WINDOW_SEC", 600)
    feedback._RATE.clear()
    feedback.check_rate_limit("1.2.3.4")
    feedback.check_rate_limit("1.2.3.4")
    with pytest.raises(feedback.FeedbackRateLimitError):
        feedback.check_rate_limit("1.2.3.4")
    feedback.check_rate_limit("9.9.9.9")  # other IP ok


def test_build_feedback_email_with_photo():
    from email.message import EmailMessage

    msg = feedback.build_feedback_email(
        email="user@example.com",
        message="Need help with my passport photo please",
        client_ip="203.0.113.9",
        user_agent="pytest",
        photo=("shot.jpg", b"\xff\xd8\xff"),
    )
    assert isinstance(msg, EmailMessage)
    assert msg["To"] == config.FEEDBACK_TO
    assert msg["Reply-To"] == "user@example.com"
    assert msg["Subject"].startswith("[GoSphoto feedback]")
    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert "user@example.com" in body
    assert "203.0.113.9" in body
    assert len(list(msg.iter_attachments())) == 1


def test_send_feedback_requires_password(monkeypatch):
    from email.message import EmailMessage

    monkeypatch.setattr(config, "SMTP_PASSWORD", "")
    msg = EmailMessage()
    msg["To"] = "mail@antonbutov.com"
    msg["From"] = "mail@antonbutov.com"
    msg.set_content("x")
    with pytest.raises(feedback.FeedbackConfigError):
        feedback.send_feedback_email(msg)


def test_send_feedback_smtp_ok(monkeypatch):
    from email.message import EmailMessage
    from unittest.mock import MagicMock, patch

    monkeypatch.setattr(config, "SMTP_PASSWORD", "secret")
    monkeypatch.setattr(config, "SMTP_HOST", "mail.antonbutov.com")
    monkeypatch.setattr(config, "SMTP_PORT", 587)
    monkeypatch.setattr(config, "SMTP_USER", "mail@antonbutov.com")
    msg = EmailMessage()
    msg["To"] = "mail@antonbutov.com"
    msg["From"] = "mail@antonbutov.com"
    msg.set_content("hi")
    smtp = MagicMock()
    smtp.__enter__.return_value = smtp
    smtp.__exit__.return_value = False
    with patch("app.feedback.smtplib.SMTP", return_value=smtp):
        feedback.send_feedback_email(msg)
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("mail@antonbutov.com", "secret")
    smtp.send_message.assert_called_once()


def _mini_feedback_app():
    import asyncio

    from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
    from fastapi.testclient import TestClient  # noqa: F401 — imported by callers

    app = FastAPI()

    @app.post("/api/feedback")
    async def post_feedback(
        request: Request,
        email: str = Form(...),
        message: str = Form(...),
        photo: UploadFile | None = File(None),
    ):
        ip = request.client.host if request.client else "unknown"
        xff = request.headers.get("x-forwarded-for")
        if xff:
            ip = xff.split(",")[0].strip() or ip
        ua = request.headers.get("user-agent", "")
        try:
            feedback.check_rate_limit(ip)
            email_n = feedback.validate_email(email)
            message_n = feedback.validate_message(message)
            raw = await photo.read() if photo is not None else None
            photo_n = feedback.validate_photo(
                photo.filename if photo else None,
                photo.content_type if photo else None,
                raw,
            )
            msg = feedback.build_feedback_email(
                email=email_n,
                message=message_n,
                client_ip=ip,
                user_agent=ua,
                photo=photo_n,
            )
            await asyncio.to_thread(feedback.send_feedback_email, msg)
        except feedback.FeedbackRateLimitError as e:
            raise HTTPException(status_code=429, detail=e.detail) from e
        except feedback.FeedbackValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail) from e
        except feedback.FeedbackConfigError as e:
            raise HTTPException(status_code=503, detail=e.detail) from e
        except feedback.FeedbackSmtpError as e:
            raise HTTPException(status_code=502, detail=e.detail) from e
        return {"ok": True}

    @app.get("/api/feedback")
    def feedback_info():
        return {
            "endpoint": "/api/feedback",
            "method": "POST",
            "fields": ["email", "message", "photo?"],
        }

    return app


def test_http_feedback_ok(monkeypatch):
    from fastapi.testclient import TestClient

    feedback._RATE.clear()
    monkeypatch.setattr(config, "SMTP_PASSWORD", "secret")
    monkeypatch.setattr(feedback, "send_feedback_email", lambda msg: None)
    client = TestClient(_mini_feedback_app())
    res = client.post(
        "/api/feedback",
        data={"email": "a@b.co", "message": "Hello, need help please"},
    )
    assert res.status_code == 200
    assert res.json() == {"ok": True}


def test_http_feedback_rate_limit(monkeypatch):
    from fastapi.testclient import TestClient

    feedback._RATE.clear()
    monkeypatch.setattr(config, "FEEDBACK_RATE_LIMIT", 1)
    monkeypatch.setattr(config, "FEEDBACK_RATE_WINDOW_SEC", 600)
    monkeypatch.setattr(config, "SMTP_PASSWORD", "secret")
    monkeypatch.setattr(feedback, "send_feedback_email", lambda msg: None)
    client = TestClient(_mini_feedback_app())
    assert (
        client.post(
            "/api/feedback",
            data={"email": "a@b.co", "message": "Hello, need help please"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/feedback",
            data={"email": "a@b.co", "message": "Hello, need help please"},
        ).status_code
        == 429
    )


def test_http_feedback_get_info():
    from fastapi.testclient import TestClient

    client = TestClient(_mini_feedback_app())
    res = client.get("/api/feedback")
    assert res.status_code == 200
    assert res.json()["method"] == "POST"
