"""Paid result → customer email (no mediapipe / main import)."""

from __future__ import annotations

from email.message import EmailMessage
from unittest.mock import MagicMock, patch

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app import config
from app import result_email as result_email_mod
from app import results


class ResultEmailBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)


def _client_ip(request: Request) -> str:
    ip = request.client.host if request.client else "unknown"
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip() or ip
    return ip


def _require_paid(result_id: str) -> None:
    if not results.is_valid_result_id(result_id) or not results.load_meta(result_id):
        raise HTTPException(status_code=404, detail="Result not found")
    if not results.is_paid(result_id):
        raise HTTPException(
            status_code=403,
            detail="Payment required. Pay via POST /api/result/{id}/pay",
        )


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "SMTP_PASSWORD", "secret")
    monkeypatch.setattr(config, "SMTP_HOST", "mail.example.com")
    monkeypatch.setattr(config, "SMTP_USER", "mail@example.com")
    monkeypatch.setattr(config, "SMTP_FROM", "mail@example.com")
    monkeypatch.setattr(config, "PUBLIC_BASE_URL", "https://gosphoto.ru")
    monkeypatch.setattr(config, "RESULT_EMAIL_RATE_LIMIT", 3)
    monkeypatch.setattr(config, "RESULT_EMAIL_RATE_WINDOW_SEC", 600)
    result_email_mod._RATE.clear()


def _mini_app():
    app = FastAPI()

    @app.post("/api/result/{result_id}/email")
    async def email_result(result_id: str, body: ResultEmailBody, request: Request):
        _require_paid(result_id)
        digital = results.load_file(result_id, "digital.jpg")
        print_jpeg = results.load_file(result_id, "print.jpg")
        if not digital or not print_jpeg:
            raise HTTPException(status_code=404, detail="Result not found")
        try:
            result_email_mod.check_rate_limit(_client_ip(request))
            email_n = result_email_mod.validate_email(body.email)
            msg = result_email_mod.build_result_email(
                email=email_n,
                result_id=result_id,
                digital_jpeg=digital,
                print_jpeg=print_jpeg,
            )
            result_email_mod.send_result_email(msg)
        except result_email_mod.FeedbackRateLimitError as e:
            raise HTTPException(status_code=429, detail=e.detail) from e
        except result_email_mod.FeedbackValidationError as e:
            raise HTTPException(status_code=e.status_code, detail=e.detail) from e
        except result_email_mod.FeedbackConfigError as e:
            raise HTTPException(status_code=503, detail=e.detail) from e
        except result_email_mod.FeedbackSmtpError as e:
            raise HTTPException(status_code=502, detail=e.detail) from e
        return {"ok": True, "email": email_n, "result_id": result_id}

    return app


def test_build_result_email_has_two_attachments():
    msg = result_email_mod.build_result_email(
        email="user@example.com",
        result_id="a" * 32,
        digital_jpeg=b"\xff\xd8\xffd",
        print_jpeg=b"\xff\xd8\xffp",
    )
    assert isinstance(msg, EmailMessage)
    assert msg["To"] == "user@example.com"
    assert "Госфото" in msg["Subject"]
    names = [a.get_filename() for a in msg.iter_attachments()]
    assert names == ["gosphoto-passport.jpg", "gosphoto-10x15.jpg"]
    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert "/result/" + ("a" * 32) in body


def test_email_forbidden_until_paid(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # Bypass preview/cv2: write result folder manually
    rid = "b" * 32
    folder = tmp_path / "results" / rid
    folder.mkdir(parents=True)
    (folder / "digital.jpg").write_bytes(b"\xff\xd8\xffd")
    (folder / "print.jpg").write_bytes(b"\xff\xd8\xffp")
    (folder / "meta.json").write_text(
        '{"result_id":"%s","paid":false}' % rid, encoding="utf-8"
    )
    client = TestClient(_mini_app())
    r = client.post(f"/api/result/{rid}/email", json={"email": "u@example.com"})
    assert r.status_code == 403


def _paid_result(tmp_path, rid: str = "c" * 32) -> str:
    folder = tmp_path / "results" / rid
    folder.mkdir(parents=True)
    (folder / "digital.jpg").write_bytes(b"\xff\xd8\xffd")
    (folder / "print.jpg").write_bytes(b"\xff\xd8\xffp")
    (folder / "meta.json").write_text(
        '{"result_id":"%s","paid":true,"payment_id":"p1"}' % rid,
        encoding="utf-8",
    )
    return rid


def test_email_ok_when_paid(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    rid = _paid_result(tmp_path)
    client = TestClient(_mini_app())
    smtp = MagicMock()
    smtp.__enter__.return_value = smtp
    smtp.__exit__.return_value = False
    with patch("app.feedback.smtplib.SMTP", return_value=smtp):
        r = client.post(
            f"/api/result/{rid}/email", json={"email": "  User@Example.com "}
        )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["email"] == "User@Example.com"
    assert body["result_id"] == rid
    smtp.login.assert_called_once()
    smtp.send_message.assert_called_once()


def test_email_rejects_bad_address(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    rid = _paid_result(tmp_path, "d" * 32)
    client = TestClient(_mini_app())
    r = client.post(f"/api/result/{rid}/email", json={"email": "not-email"})
    assert r.status_code == 400


def test_email_rate_limit(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "RESULT_EMAIL_RATE_LIMIT", 2)
    rid = _paid_result(tmp_path, "e" * 32)
    client = TestClient(_mini_app())
    smtp = MagicMock()
    smtp.__enter__.return_value = smtp
    smtp.__exit__.return_value = False
    with patch("app.feedback.smtplib.SMTP", return_value=smtp):
        assert (
            client.post(
                f"/api/result/{rid}/email", json={"email": "a@example.com"}
            ).status_code
            == 200
        )
        assert (
            client.post(
                f"/api/result/{rid}/email", json={"email": "b@example.com"}
            ).status_code
            == 200
        )
        r = client.post(
            f"/api/result/{rid}/email", json={"email": "c@example.com"}
        )
    assert r.status_code == 429
