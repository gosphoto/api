# Gosphoto API — photo gate + Riverflow white bg + crop

Бэкенд для https://gosphoto.ru

- `POST /api/validate` — gate (MediaPipe Face Landmarker)
- `POST /api/process` — gate → **Riverflow v2.5 Pro** (`background_mode=solid` `#FFFFFF`) → 35×45 crop → **лист 10×15** → `result_id` (без полного base64); fallback — local cutout
- `GET /api/result/{id}` — meta + preview URLs; `paid` / цена
- `POST /api/result/{id}/pay` — Tochka checkout (70 ₽) → unlock download
- `POST /api/payments/tochka/webhook` — webhook Точки
- `GET /api/result/{id}/digital.jpg|print.jpg` — только после оплаты
- `POST /api/result/{id}/email` — JSON `{ "email" }` → SMTP с вложениями (только после оплаты)
- `POST /api/edit` — только белый фон; может использовать OpenRouter если `EDIT_BACKEND=openrouter`
- `POST /api/feedback` — multipart: `email`, `message`, optional `photo` → SMTP to `FEEDBACK_TO`
- `GET /health`

Чеклист оплаты: [docs/payments-tochka.md](docs/payments-tochka.md)

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

Push / PR → GitHub Actions: **pytest** (crop regression: лысый / высокая укладка / объёмные волосы; upright regression: перевёрнутое селфи) → deploy на `main` только если тесты зелёные.

Локально:

```bash
pip install -r requirements.txt pytest
export GATE_MODEL_PATH=$PWD/models/face_landmarker.task PYTHONPATH=$PWD
pytest tests/ -v
```

## Secrets

https://github.com/gosphoto/api/settings/secrets/actions

| Secret | Обязателен | Назначение |
|--------|------------|------------|
| `DEPLOY_SSH_PRIVATE_KEY` | да | SSH на VPS |
| `DEPLOY_USER` | да | SSH user (обычно `root`) |
| `OPENROUTER_API_KEY` | да | Riverflow / OpenRouter image edit |
| `RIVERFLOW_MODEL` | нет | default `black-forest-labs/flux.2-pro` (Riverflow Fast отвергнут — см. `docs/riverflow-fast-vs-pro.md`) |
| `RIVERFLOW_BG_MODE` / `RIVERFLOW_BG_HEX` | нет | `solid` / `#FFFFFF` |
| `OPENROUTER_IMAGE_MODEL` | нет | legacy model for non-Riverflow `/api/edit` |

Для `/api/feedback` — GitHub Secrets (пишет deploy в `/opt/gosphoto-api/.env`):

| Secret | Default если пусто | Назначение |
|--------|-------------------|------------|
| `SMTP_HOST` | `mail.antonbutov.com` | SMTP |
| `SMTP_PORT` | `587` | STARTTLS |
| `SMTP_USER` | `mail@antonbutov.com` | auth |
| `SMTP_PASSWORD` | _(required)_ | auth |
| `SMTP_FROM` / `FEEDBACK_TO` | `mail@antonbutov.com` | From / To |
| `TOCHKA_ACCESS_TOKEN` | _(required for live pay)_ | JWT эквайринга Точки |
| `TOCHKA_CUSTOMER_CODE` | optional | иначе resolve из API Точки |
| `TOCHKA_MERCHANT_ID` | optional | merchantId |

Скопируй `DEPLOY_*` из [gosphoto/landing](https://github.com/gosphoto/landing/settings/secrets/actions).

Webhook Точки: `https://gosphoto.ru/api/payments/tochka/webhook`

## Local

```bash
cp .env.example .env
docker compose up --build
curl -F file=@selfie.jpg http://127.0.0.1:8091/api/validate
curl -F file=@selfie.jpg "http://127.0.0.1:8091/api/process?format=jpeg" -o out.jpg
```
