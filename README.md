# Gosphoto API — photo gate + local white bg + crop

Бэкенд для https://gosphoto.ru

- `POST /api/validate` — gate (MediaPipe Face Landmarker)
- `POST /api/process` — gate → **локальный** белый фон (cutout) → 35×45 crop  
  (без generative OpenRouter — лицо из исходных пикселей)
- `POST /api/edit` — только белый фон; может использовать OpenRouter если `EDIT_BACKEND=openrouter`
- `GET /health`

Требования кадра: 35×45 мм @ 300 dpi (413×531), лицо ~70–80%, верхнее поле ~0.11, фон `#FFFFFF`.

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
| `OPENROUTER_API_KEY` | нет | только для опционального `/api/edit` |
| `OPENROUTER_IMAGE_MODEL` | нет | default image model for `/api/edit` |

Скопируй `DEPLOY_*` из [gosphoto/landing](https://github.com/gosphoto/landing/settings/secrets/actions).

## Local

```bash
cp .env.example .env
docker compose up --build
curl -F file=@selfie.jpg http://127.0.0.1:8091/api/validate
curl -F file=@selfie.jpg "http://127.0.0.1:8091/api/process?format=jpeg" -o out.jpg
```
