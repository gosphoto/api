# Feedback / contacts page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `/contacts` feedback form (email + message + optional photo) and `POST /api/feedback` that emails `mail@antonbutov.com` via SMTP `mail.antonbutov.com`.

**Architecture:** Static `contacts.html` on the landing posts multipart to FastAPI. A focused `app/feedback.py` validates input, rate-limits by IP, builds a MIME message, and sends it with stdlib `smtplib` in a worker thread. Header/footer «Контакты» point at `/contacts` instead of mailto.

**Tech Stack:** FastAPI, python-multipart, stdlib `smtplib`/`email`, static HTML/CSS/JS, nginx.

**Spec:** `docs/superpowers/specs/2026-08-08-feedback-contacts-design.md`

## Global Constraints

- To address: `mail@antonbutov.com` (env `FEEDBACK_TO`)
- SMTP host: `mail.antonbutov.com` port `587` STARTTLS
- Fields: required `email` + `message` (10–4000 chars); optional one photo JPEG/PNG/WebP ≤5 MB
- Rate-limit: 5 requests / 600s / client IP (in-process)
- No captcha, no disk persistence of feedback, no multi-photo
- Repos: API work in `gosphoto-api`; UI/nginx in `gosphoto-landing`
- Do not commit SMTP passwords

## File map

| File | Role |
|------|------|
| `gosphoto-api/app/config.py` | SMTP + feedback env defaults |
| `gosphoto-api/app/feedback.py` | validate, rate-limit, build MIME, send SMTP |
| `gosphoto-api/app/main.py` | `POST/GET /api/feedback` |
| `gosphoto-api/tests/test_feedback.py` | unit + mini-app HTTP tests |
| `gosphoto-api/README.md` | document endpoint + env |
| `gosphoto-landing/contacts.html` | feedback page |
| `gosphoto-landing/css/styles.css` | form styles |
| `gosphoto-landing/index.html` | contact links → `/contacts` |
| `gosphoto-landing/result.html` | contact links → `/contacts` |
| `gosphoto-landing/deploy/gosphoto.ru.nginx.conf` | `location = /contacts` |
| `gosphoto-landing/deploy/gosphoto.ru.nginx.http.conf` | same for HTTP vhost if present |

---

### Task 1: API — `feedback.py` validation + rate-limit + SMTP send

**Repo:** gosphoto-api

**Files:**
- Create: `app/feedback.py`
- Modify: `app/config.py` (append SMTP/feedback settings)
- Test: `tests/test_feedback.py`

**Interfaces:**
- Consumes: `app.config` SMTP/feedback constants
- Produces:
  - `validate_email(value: str) -> str` (normalized or raises `FeedbackValidationError`)
  - `validate_message(value: str) -> str`
  - `validate_photo(filename: str | None, content_type: str | None, data: bytes | None) -> tuple[str, bytes] | None`
  - `check_rate_limit(ip: str) -> None` (raises `FeedbackRateLimitError`)
  - `build_feedback_email(*, email: str, message: str, client_ip: str, user_agent: str, photo: tuple[str, bytes] | None) -> EmailMessage`
  - `send_feedback_email(msg: EmailMessage) -> None` (raises `FeedbackSmtpError` / `FeedbackConfigError`)
  - Exceptions: `FeedbackValidationError(status_code: int, detail: str)`, `FeedbackRateLimitError`, `FeedbackConfigError`, `FeedbackSmtpError`

- [ ] **Step 1: Write failing tests for validation + rate-limit**

Create `tests/test_feedback.py`:

```python
import time

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
```

- [ ] **Step 2: Run tests — expect FAIL (module missing)**

Run: `cd /Users/antonbutov/Documents/MYPROJECTS/gosphoto-api && python -m pytest tests/test_feedback.py -v`  
Expected: `ModuleNotFoundError` or import error for `app.feedback`

- [ ] **Step 3: Add config keys**

Append to `app/config.py`:

```python
SMTP_HOST = os.getenv("SMTP_HOST", "mail.antonbutov.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "mail@antonbutov.com").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", "mail@antonbutov.com").strip()
FEEDBACK_TO = os.getenv("FEEDBACK_TO", "mail@antonbutov.com").strip()
FEEDBACK_RATE_LIMIT = int(os.getenv("FEEDBACK_RATE_LIMIT", "5"))
FEEDBACK_RATE_WINDOW_SEC = int(os.getenv("FEEDBACK_RATE_WINDOW_SEC", "600"))
FEEDBACK_MAX_PHOTO_BYTES = int(os.getenv("FEEDBACK_MAX_PHOTO_BYTES", str(5 * 1024 * 1024)))
FEEDBACK_MAX_MESSAGE_CHARS = int(os.getenv("FEEDBACK_MAX_MESSAGE_CHARS", "4000"))
FEEDBACK_MIN_MESSAGE_CHARS = int(os.getenv("FEEDBACK_MIN_MESSAGE_CHARS", "10"))
```

- [ ] **Step 4: Implement `app/feedback.py` (validation + rate-limit first)**

```python
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
    ext = _ALLOWED_PHOTO[ct if ct != "image/jpg" else "image/jpeg"]
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
```

- [ ] **Step 5: Run validation/rate tests — expect PASS**

Run: `python -m pytest tests/test_feedback.py -v`  
Expected: all current tests PASS

- [ ] **Step 6: Add tests for build + send (mocked SMTP)**

Append to `tests/test_feedback.py`:

```python
from email.message import EmailMessage
from unittest.mock import MagicMock, patch


def test_build_feedback_email_with_photo():
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
    monkeypatch.setattr(config, "SMTP_PASSWORD", "")
    msg = EmailMessage()
    msg["To"] = "mail@antonbutov.com"
    msg["From"] = "mail@antonbutov.com"
    msg.set_content("x")
    with pytest.raises(feedback.FeedbackConfigError):
        feedback.send_feedback_email(msg)


def test_send_feedback_smtp_ok(monkeypatch):
    monkeypatch.setattr(config, "SMTP_PASSWORD", "secret")
    monkeypatch.setattr(config, "SMTP_HOST", "mail.antonbutov.com")
    monkeypatch.setattr(config, "SMTP_PORT", 587)
    monkeypatch.setattr(config, "SMTP_USER", "mail@antonbutov.com")
    msg = EmailMessage()
    msg["To"] = "mail@antonbutov.com"
    msg["From"] = "mail@antonbutov.com"
    msg.set_content("hi")
    smtp = MagicMock()
    with patch("app.feedback.smtplib.SMTP", return_value=smtp) as ctor:
        smtp.__enter__.return_value = smtp
        # support both context-manager and non-CM style
        ctor.return_value = smtp
        feedback.send_feedback_email(msg)
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("mail@antonbutov.com", "secret")
    smtp.send_message.assert_called_once()
```

- [ ] **Step 7: Implement build + send in `app/feedback.py`**

```python
def build_feedback_email(
    *,
    email: str,
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
                f"IP: {client_ip}",
                f"User-Agent: {user_agent}",
                "",
                message,
            ]
        )
    )
    if photo:
        filename, data = photo
        maintype, subtype = ("image", "jpeg")
        lower = filename.lower()
        if lower.endswith(".png"):
            subtype = "png"
        elif lower.endswith(".webp"):
            subtype = "webp"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
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
```

Adjust the mock test if implementation uses `with SMTP(...)` — `MagicMock` as context manager:

```python
smtp = MagicMock()
smtp.__enter__.return_value = smtp
smtp.__exit__.return_value = False
with patch("app.feedback.smtplib.SMTP", return_value=smtp):
    feedback.send_feedback_email(msg)
```

- [ ] **Step 8: Run full `tests/test_feedback.py` — PASS**

Run: `python -m pytest tests/test_feedback.py -v`  
Expected: PASS

- [ ] **Step 9: Commit (api)**

```bash
cd /Users/antonbutov/Documents/MYPROJECTS/gosphoto-api
git add app/config.py app/feedback.py tests/test_feedback.py
git commit -m "$(cat <<'EOF'
Add feedback email helpers with validation and SMTP send.

EOF
)"
```

---

### Task 2: API — wire `POST/GET /api/feedback`

**Repo:** gosphoto-api

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_feedback.py` (HTTP mini-app)
- Modify: `README.md` (document endpoint + env)

**Interfaces:**
- Consumes: all Task 1 functions/exceptions
- Produces: HTTP routes `POST /api/feedback`, `GET /api/feedback`

- [ ] **Step 1: Write failing HTTP tests (mini-app, no mediapipe)**

Append to `tests/test_feedback.py`:

```python
import asyncio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.testclient import TestClient


def _mini_feedback_app():
    app = FastAPI()

    @app.post("/api/feedback")
    async def post_feedback(
        request: Request,
        email: str = Form(...),
        message: str = Form(...),
        photo: UploadFile | None = File(None),
    ):
        ip = request.client.host if request.client else "unknown"
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
    feedback._RATE.clear()
    monkeypatch.setattr(config, "FEEDBACK_RATE_LIMIT", 1)
    monkeypatch.setattr(config, "FEEDBACK_RATE_WINDOW_SEC", 600)
    monkeypatch.setattr(config, "SMTP_PASSWORD", "secret")
    monkeypatch.setattr(feedback, "send_feedback_email", lambda msg: None)
    client = TestClient(_mini_feedback_app())
    assert client.post(
        "/api/feedback",
        data={"email": "a@b.co", "message": "Hello, need help please"},
    ).status_code == 200
    assert client.post(
        "/api/feedback",
        data={"email": "a@b.co", "message": "Hello, need help please"},
    ).status_code == 429


def test_http_feedback_get_info():
    client = TestClient(_mini_feedback_app())
    res = client.get("/api/feedback")
    assert res.status_code == 200
    assert res.json()["method"] == "POST"
```

- [ ] **Step 2: Run new HTTP tests against real `main` or implement route**

Prefer implementing the same handler body in `app/main.py` (not only in the test helper). After wiring, either keep the mini-app tests (mirrors route logic — acceptable like `test_results.py`) **or** import `app.main.app` if mediapipe warmup is skipped in tests. Follow existing pattern: **mini-app in tests**, then copy the handler into `main.py` identically.

- [ ] **Step 3: Wire routes in `app/main.py`**

Add imports:

```python
import asyncio
from fastapi import Form, Request

from . import feedback as feedback_mod
```

Add handlers (place near other `/api/*` routes):

```python
@app.post("/api/feedback")
async def post_feedback(
    request: Request,
    email: str = Form(...),
    message: str = Form(...),
    photo: UploadFile | None = File(None),
):
    ip = request.client.host if request.client else "unknown"
    # Prefer X-Forwarded-For first hop when behind nginx
    xff = request.headers.get("x-forwarded-for")
    if xff:
        ip = xff.split(",")[0].strip() or ip
    ua = request.headers.get("user-agent", "")
    try:
        feedback_mod.check_rate_limit(ip)
        email_n = feedback_mod.validate_email(email)
        message_n = feedback_mod.validate_message(message)
        raw = await photo.read() if photo is not None else None
        photo_n = feedback_mod.validate_photo(
            photo.filename if photo else None,
            photo.content_type if photo else None,
            raw,
        )
        msg = feedback_mod.build_feedback_email(
            email=email_n,
            message=message_n,
            client_ip=ip,
            user_agent=ua,
            photo=photo_n,
        )
        await asyncio.to_thread(feedback_mod.send_feedback_email, msg)
    except feedback_mod.FeedbackRateLimitError as e:
        raise HTTPException(status_code=429, detail=e.detail) from e
    except feedback_mod.FeedbackValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e
    except feedback_mod.FeedbackConfigError as e:
        raise HTTPException(status_code=503, detail=e.detail) from e
    except feedback_mod.FeedbackSmtpError as e:
        raise HTTPException(status_code=502, detail=e.detail) from e
    return {"ok": True}


@app.get("/api/feedback")
def feedback_info():
    return {
        "endpoint": "/api/feedback",
        "method": "POST",
        "fields": {
            "email": "required, reply-to",
            "message": "required, 10–4000 chars",
            "photo": "optional, JPEG/PNG/WebP ≤5MB",
        },
        "to": config.FEEDBACK_TO,
    }
```

Also include `/api/feedback` in `process_info` endpoints map if that dict is maintained, and bump health `version` patch (e.g. `0.7.0` → next) with a short note that feedback exists.

Update mini-app in tests to also honor `X-Forwarded-For` the same way **or** document that only production `main.py` does — prefer **same logic in both**.

- [ ] **Step 4: Update README**

Document:

```markdown
- `POST /api/feedback` — multipart: `email`, `message`, optional `photo` → SMTP to FEEDBACK_TO
```

Env table rows for `SMTP_*` and `FEEDBACK_*`.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_feedback.py -v`  
Expected: PASS

- [ ] **Step 6: Commit (api)**

```bash
git add app/main.py tests/test_feedback.py README.md
git commit -m "$(cat <<'EOF'
Expose POST /api/feedback for contact form email.

EOF
)"
```

---

### Task 3: Landing — `contacts.html` + CSS + nginx

**Repo:** gosphoto-landing

**Files:**
- Create: `contacts.html`
- Modify: `css/styles.css`
- Modify: `deploy/gosphoto.ru.nginx.conf`
- Modify: `deploy/gosphoto.ru.nginx.http.conf`

**Interfaces:**
- Consumes: `POST /api/feedback` multipart contract from Task 2
- Produces: page at `/contacts` (nginx) and `/contacts.html`

- [ ] **Step 1: Add nginx locations**

In both nginx confs, after the `/result/` location block:

```nginx
    location = /contacts {
        try_files /contacts.html =404;
        add_header Cache-Control "no-cache";
    }
```

- [ ] **Step 2: Add CSS for the form**

Append to `css/styles.css`:

```css
.contacts-page-main {
  max-width: var(--max);
  margin: 0 auto;
  padding: 2rem 1.5rem 4rem;
}

.contacts-panel {
  max-width: 36rem;
}

.contacts-panel h1 {
  margin: 0 0 0.5rem;
  font-size: clamp(1.8rem, 3vw, 2.4rem);
  letter-spacing: -0.03em;
}

.contacts-lead {
  margin: 0 0 1.5rem;
  color: var(--muted);
}

.contacts-form {
  display: grid;
  gap: 1rem;
}

.contacts-field {
  display: grid;
  gap: 0.35rem;
}

.contacts-field label {
  font-weight: 700;
  font-size: 0.92rem;
}

.contacts-field input,
.contacts-field textarea {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 0.9rem;
  padding: 0.85rem 1rem;
  font: inherit;
  background: var(--surface);
  color: var(--ink);
}

.contacts-field textarea {
  min-height: 9rem;
  resize: vertical;
}

.contacts-photo-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  align-items: center;
}

.contacts-photo-name {
  color: var(--muted);
  font-size: 0.9rem;
}

.contacts-status {
  min-height: 1.4rem;
  font-weight: 600;
  font-size: 0.95rem;
}

.contacts-status.is-error {
  color: #b42318;
}

.contacts-status.is-ok {
  color: #067647;
}

.contacts-form .btn[disabled] {
  opacity: 0.65;
  cursor: wait;
}
```

- [ ] **Step 3: Create `contacts.html`**

Use the same head/metrika/header/footer chrome as `result.html`. Main content:

```html
<main class="contacts-page-main">
  <section class="contacts-panel">
    <h1>Обратная связь</h1>
    <p class="contacts-lead">
      Напишите нам — ответим на ваш email в течение дня.
    </p>
    <form class="contacts-form" id="feedback-form" novalidate>
      <div class="contacts-field">
        <label for="feedback-email">Email для ответа</label>
        <input id="feedback-email" name="email" type="email" required maxlength="254" autocomplete="email" />
      </div>
      <div class="contacts-field">
        <label for="feedback-message">Сообщение</label>
        <textarea id="feedback-message" name="message" required minlength="10" maxlength="4000"></textarea>
      </div>
      <div class="contacts-field">
        <label for="feedback-photo">Фото (необязательно)</label>
        <div class="contacts-photo-row">
          <input id="feedback-photo" name="photo" type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" />
          <button type="button" class="btn btn-dark" id="feedback-photo-clear" hidden>Убрать фото</button>
        </div>
        <span class="contacts-photo-name" id="feedback-photo-name"></span>
      </div>
      <button class="btn btn-primary" type="submit" id="feedback-submit">Отправить</button>
      <p class="contacts-status" id="feedback-status" role="status" aria-live="polite"></p>
    </form>
  </section>
</main>
```

Header contact `href="/contacts"`. Inline script:

```javascript
(function () {
  const form = document.getElementById("feedback-form");
  const email = document.getElementById("feedback-email");
  const message = document.getElementById("feedback-message");
  const photo = document.getElementById("feedback-photo");
  const photoName = document.getElementById("feedback-photo-name");
  const photoClear = document.getElementById("feedback-photo-clear");
  const submit = document.getElementById("feedback-submit");
  const status = document.getElementById("feedback-status");

  function setStatus(text, kind) {
    status.textContent = text || "";
    status.classList.remove("is-error", "is-ok");
    if (kind) status.classList.add(kind);
  }

  photo.addEventListener("change", () => {
    const f = photo.files && photo.files[0];
    photoName.textContent = f ? f.name : "";
    photoClear.hidden = !f;
  });
  photoClear.addEventListener("click", () => {
    photo.value = "";
    photoName.textContent = "";
    photoClear.hidden = true;
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    setStatus("");
    const body = new FormData();
    body.append("email", email.value.trim());
    body.append("message", message.value.trim());
    if (photo.files && photo.files[0]) body.append("photo", photo.files[0]);
    submit.disabled = true;
    try {
      const res = await fetch("/api/feedback", { method: "POST", body });
      if (!res.ok) {
        let detail = "Не удалось отправить. Попробуйте позже.";
        try {
          const j = await res.json();
          if (j && j.detail) detail = typeof j.detail === "string" ? j.detail : detail;
        } catch (_) {}
        setStatus(detail, "is-error");
        if (typeof ym === "function") ym(111303098, "reachGoal", "feedback_fail");
        return;
      }
      form.reset();
      photoName.textContent = "";
      photoClear.hidden = true;
      setStatus("Сообщение отправлено. Мы ответим на ваш email.", "is-ok");
      if (typeof ym === "function") ym(111303098, "reachGoal", "feedback_ok");
    } catch (_) {
      setStatus("Сеть недоступна. Попробуйте позже.", "is-error");
      if (typeof ym === "function") ym(111303098, "reachGoal", "feedback_fail");
    } finally {
      submit.disabled = false;
    }
  });
})();
```

- [ ] **Step 4: Manual local check (optional)**

Open `contacts.html` via any static server; verify layout. API call can wait for deploy or local API with SMTP mocked.

- [ ] **Step 5: Commit (landing)**

```bash
cd /Users/antonbutov/Documents/MYPROJECTS/gosphoto-landing
git add contacts.html css/styles.css deploy/gosphoto.ru.nginx.conf deploy/gosphoto.ru.nginx.http.conf
git commit -m "$(cat <<'EOF'
Add /contacts feedback form page and nginx route.

EOF
)"
```

---

### Task 4: Landing — point «Контакты» links to `/contacts`

**Repo:** gosphoto-landing

**Files:**
- Modify: `index.html` (header + footer mailto → `/contacts`)
- Modify: `result.html` (same)

- [ ] **Step 1: Update links**

Replace:

```html
href="mailto:hello@gosphoto.ru"
```

on contact entry points with:

```html
href="/contacts"
```

Keep visible label `hello@gosphoto.ru` / «Контакты» / «Ответим в течение дня». Keep `data-metrika-goal="click_contact_email"` (or rename goal to `click_contacts` consistently on all three pages — pick one name and use it everywhere).

- [ ] **Step 2: Commit (landing)**

```bash
git add index.html result.html
git commit -m "$(cat <<'EOF'
Route contact links to /contacts feedback page.

EOF
)"
```

---

### Task 5: Deploy + smoke

**Repos:** gosphoto-api + gosphoto-landing (+ server `.env`)

- [ ] **Step 1: Set SMTP secrets on API host**

On `80.87.196.33` in `/opt/gosphoto-api/.env` (do not commit):

```
SMTP_HOST=mail.antonbutov.com
SMTP_PORT=587
SMTP_USER=mail@antonbutov.com
SMTP_PASSWORD=<from mail credentials>
SMTP_FROM=mail@antonbutov.com
FEEDBACK_TO=mail@antonbutov.com
```

Obtain password from existing mail deployment secrets (same account as mail MCP) — never paste into git/chat logs if avoidable; set via SSH editor.

- [ ] **Step 2: Push + deploy API**

```bash
cd /Users/antonbutov/Documents/MYPROJECTS/gosphoto-api
git push origin main
```

Or rsync/compose per `README.md` if CI is not used for API:

```bash
rsync -az --delete --exclude '.env' --exclude 'rejecteds/' --exclude 'pairs/' --exclude 'results/' \
  ./ root@80.87.196.33:/opt/gosphoto-api/
ssh root@80.87.196.33 'cd /opt/gosphoto-api && docker compose -f docker-compose.prod33.yml up -d --build'
```

- [ ] **Step 3: Push + deploy landing**

```bash
cd /Users/antonbutov/Documents/MYPROJECTS/gosphoto-landing
git push origin main
```

Confirm Actions reloads nginx so `location = /contacts` is live.

- [ ] **Step 4: Smoke**

1. `curl -sS https://gosphoto.ru/api/feedback` → JSON help  
2. Open `https://gosphoto.ru/contacts`  
3. Submit email + message (+ small JPEG)  
4. Confirm inbox `mail@antonbutov.com` has `[GoSphoto feedback] …` with Reply-To = submitted email  
5. Click header/footer «Контакты» from `/` → lands on `/contacts`

- [ ] **Step 5: Report**

Record: API SHA, landing SHA, smoke PASS/FAIL, message UID if checked via mail MCP.

## Done when

- `/contacts` form works on prod
- `POST /api/feedback` delivers mail to `mail@antonbutov.com` via `mail.antonbutov.com`
- Contact entry points no longer open mailto
- `tests/test_feedback.py` green locally

## Spec coverage checklist

| Spec item | Task |
|-----------|------|
| `/contacts` page + form fields | Task 3 |
| nginx `/contacts` | Task 3 |
| Links from index/result | Task 4 |
| `POST /api/feedback` multipart | Task 2 |
| Validation + 400/413 | Task 1–2 |
| Rate-limit 429 | Task 1–2 |
| SMTP STARTTLS + Reply-To + attachment | Task 1 |
| Env config | Task 1 + 5 |
| GET help | Task 2 |
| Deploy + smoke | Task 5 |
| No captcha / no multi-photo | Global constraints |
