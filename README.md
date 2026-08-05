# Gosphoto API — photo gate + OpenRouter edit

Бэкенд для https://gosphoto.ru

- `POST /api/validate` — gate (MediaPipe Face Landmarker)
- `POST /api/process` — gate → OpenRouter edit → local 35×45 crop
- `GET /health`

## Deploy

VPS `91.207.75.72` → `/opt/gosphoto-api` (Docker `gosphoto-gate`, `127.0.0.1:8091`).  
Nginx на лендинге проксирует `/api/` и `/health`.

Push в `main` → GitHub Actions deploy.

## Secrets

https://github.com/gosphoto/api/settings/secrets/actions

| Secret | Обязателен | Назначение |
|--------|------------|------------|
| `DEPLOY_SSH_PRIVATE_KEY` | да | SSH на VPS |
| `DEPLOY_USER` | да | SSH user (обычно `root`) |
| `OPENROUTER_API_KEY` | да для `/api/process` | OpenRouter |
| `OPENROUTER_IMAGE_MODEL` | нет | default `google/gemini-2.5-flash-image` |

Скопируй `DEPLOY_*` из [gosphoto/landing](https://github.com/gosphoto/landing/settings/secrets/actions).

## Local

```bash
cp .env.example .env   # вставь OPENROUTER_API_KEY
docker compose up --build
curl -F file=@selfie.jpg http://127.0.0.1:8091/api/validate
```
