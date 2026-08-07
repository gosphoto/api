# Gosphoto API — photo gate + local white bg + crop

Бэкенд для https://gosphoto.ru

- `POST /api/validate` — gate (MediaPipe Face Landmarker)
- `POST /api/process` — gate → белый фон → 35×45 crop → **лист 10×15 (4 фото)**  
  JSON: `image_base64` (Госуслуги) + `print_sheet.image_base64` (печать)
- `POST /api/edit` — только белый фон; может использовать OpenRouter если `EDIT_BACKEND=openrouter`
- `GET /health`

Требования кадра ([FMS §34.3 / rg.ru](https://rg.ru/documents/2011/08/22/pasport-dok.html)):

| Параметр | Норма |
|----------|--------|
| Размер | **35×45 мм** |
| DPI | **≥600** → **827×1063** px |
| Овал лица | **≥80%** высоты кадра |
| Голова | длина **32–36 мм**, ширина **18–25 мм** |
| Файл | **JPEG**, **≤300 КБ** |
| Фон | `#FFFFFF` |

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
