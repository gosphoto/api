# Feedback / contacts page — Design

**Date:** 2026-08-08  
**Repos:** gosphoto-api + gosphoto-landing  
**Status:** approved

## Problem

«Контакты» on gosphoto.ru open `mailto:hello@gosphoto.ru`. There is no in-product way to send a message with an optional photo. Support mail should land in `mail@antonbutov.com` via `mail.antonbutov.com` SMTP.

## Goal

- Dedicated contacts page that is a feedback form (text + reply email + optional one photo).
- Header/footer «Контакты» and the header email block link to that page (not mailto).
- New REST endpoint sends the message as email to `mail@antonbutov.com`.

## Non-goals

- Captcha / honeypot
- Multiple photo attachments
- Separate legal/requisites page
- Mail queue / retries beyond a single SMTP attempt
- Changing photo process pipeline
- Storing feedback on disk (email is the sink)

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Entry | A — «Контакты» → feedback page |
| Fields | B — message + user email + optional photo |
| Photos | A — at most one |
| Anti-spam | A — length limits + IP rate-limit, no captcha |
| Delivery | 1 — SMTP from FastAPI → `mail.antonbutov.com` |

## Architecture

```
[header/footer Контакты]
        ↓
/contacts  (contacts.html)
        ↓  POST multipart /api/feedback
gosphoto-gate (80.87.196.33:8111)
        ↓  SMTP STARTTLS → mail.antonbutov.com:587
mail@antonbutov.com  (Reply-To = user email)
```

Nginx on `91.207.75.72` already proxies `/api/` — no nginx change for the new path. Static page deploys with the landing.

## API (gosphoto-api)

### `POST /api/feedback`

Multipart fields:

| Field | Required | Rules |
|-------|----------|-------|
| `email` | yes | valid email, ≤254 chars |
| `message` | yes | trim; 10–4000 chars |
| `photo` | no | one file; JPEG/PNG/WebP; ≤5 MB (`FEEDBACK_MAX_PHOTO_BYTES`) |

Behavior:

1. Validate fields; reject bad types / empty / oversized.
2. Rate-limit: **5 requests / 10 minutes / client IP** (in-process dict; sufficient for one container). Exceed → `429`.
3. Build MIME email:
   - From: `SMTP_FROM` (default `mail@antonbutov.com`)
   - To: `FEEDBACK_TO` (default `mail@antonbutov.com`)
   - Reply-To: user `email`
   - Subject: `[GoSphoto feedback] {first ~60 chars of message}`
   - Text body: user email, full message, client IP, User-Agent
   - Optional photo as one attachment (sanitized filename)
4. Send via SMTP STARTTLS to `SMTP_HOST`:`SMTP_PORT` with `SMTP_USER` / `SMTP_PASSWORD`.
5. Return `{"ok": true}` on success.

Errors:

| Status | When |
|--------|------|
| 400 | invalid email/message/photo type |
| 413 | photo too large |
| 429 | rate-limit |
| 502 | SMTP failure / misconfiguration |
| 503 | SMTP credentials missing |

### `GET /api/feedback`

Short JSON help (same pattern as other GET stubs on process/edit/crop).

### Config (env, not in git)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SMTP_HOST` | `mail.antonbutov.com` | SMTP host |
| `SMTP_PORT` | `587` | STARTTLS port |
| `SMTP_USER` | `mail@antonbutov.com` | auth user |
| `SMTP_PASSWORD` | _(required)_ | auth password |
| `SMTP_FROM` | `mail@antonbutov.com` | From |
| `FEEDBACK_TO` | `mail@antonbutov.com` | To |
| `FEEDBACK_RATE_LIMIT` | `5` | max hits per window |
| `FEEDBACK_RATE_WINDOW_SEC` | `600` | window seconds |
| `FEEDBACK_MAX_PHOTO_BYTES` | `5242880` | 5 MB |
| `FEEDBACK_MAX_MESSAGE_CHARS` | `4000` | message max |
| `FEEDBACK_MIN_MESSAGE_CHARS` | `10` | message min |

Implementation sketch: `app/feedback.py` (validate + rate-limit + send), route wired in `main.py`. Prefer stdlib `smtplib` + `email.message` in a thread/`asyncio.to_thread` to avoid a new heavy dependency; `aiosmtplib` acceptable if preferred for clarity.

## Landing (gosphoto-landing)

### Page

- File: `contacts.html`
- Public URL: **`/contacts`** via nginx:
  ```nginx
  location = /contacts {
      try_files /contacts.html =404;
      add_header Cache-Control "no-cache";
  }
  ```
  (same pattern as `/result/<id>` → `result.html`). Direct `/contacts.html` also works via `try_files $uri`.
- Reuse site chrome (header/footer), Manrope, existing CSS variables / buttons.
- Main: title «Обратная связь», short lead, form (email, textarea, one photo picker with preview/clear, submit).
- States: button loading, success message, inline error for 4xx/5xx.
- Metrika goals on submit success / fail (same style as existing `data-metrika-goal`).

### Link updates

Replace `mailto:hello@gosphoto.ru` contact entry points with **`/contacts`**:

- `index.html` — header contact block + footer «Контакты»
- `result.html` — same
- `contacts.html` — header email block points to self or stays as page identity

Keep visual copy (e.g. show `hello@gosphoto.ru` / «Ответим в течение дня») but `href="/contacts"`.

### Client

- `fetch("/api/feedback", { method: "POST", body: FormData })`
- No base64; multipart only
- Disable double-submit while in flight

## Deploy

1. Set SMTP env on API host `/opt/gosphoto-api/.env`, rebuild/restart `gosphoto-gate`.
2. Ship landing (`contacts.html` + link/CSS/JS + nginx `location = /contacts`) via existing GitHub Actions rsync / install script.
3. Smoke: open `https://gosphoto.ru/contacts`, submit with a small JPEG → inbox `mail@antonbutov.com`.

Note: `client_max_body_size 20m` on `/api/` already covers the 5 MB feedback photo.

## Testing

- Unit: email/message/photo validation; rate-limit trips on 6th hit.
- Manual: end-to-end form → SMTP → inbox (channel already verified with a test message on 2026-08-07).

## Out of scope follow-ups

- Persist feedback copies on disk
- Multi-instance rate-limit (Redis)
- SmartCaptcha if spam appears
- Legal requisites page separate from feedback
