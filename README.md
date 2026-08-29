# Gosphoto API — photo gate + Riverflow white bg + crop

Бэкенд для https://gosphoto.ru

- `POST /api/validate` — gate (MediaPipe Face Landmarker)
- `POST /api/process` — gate → **Gemini Flash Image** (белый фон + костюм) → 35×45 crop → **лист 10×15** → `result_id`. Второй Gemini после кропа (`POST_CROP_CLEANUP`) **выключен**: раздувает овал лица, [замер](docs/post-crop-cleanup-face-widen.md). Fallback — local cutout
- `GET /api/result/{id}` — meta + preview URLs; `paid` / цена
- `POST /api/result/{id}/pay` — Tochka checkout (400 ₽) → unlock download
- `POST /api/payments/tochka/webhook` — webhook Точки
- `GET /api/result/{id}/digital.jpg|print.jpg` — только после оплаты
- `POST /api/result/{id}/email` — JSON `{ "email" }` → SMTP с вложениями (только после оплаты)
- `POST /api/edit` — только белый фон; может использовать OpenRouter если `EDIT_BACKEND=openrouter`
- `POST /api/feedback` — multipart: `full_name`, `email`, `message`, `photo` → SMTP to `FEEDBACK_TO`
- `GET /health`

Чеклист оплаты: [docs/payments-tochka.md](docs/payments-tochka.md)

Требования кадра ([FMS §34.3 / rg.ru](https://rg.ru/documents/2011/08/22/pasport-dok.html)):

| Параметр | Норма |
|----------|--------|
| Размер | **35×45 мм** |
| DPI | **≥600** → **827×1063** px |
| Голова | длина **32–36 мм**, ширина **18–25 мм** (≈70–80% кадра) |
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
| `RIVERFLOW_MODEL` | нет | default Gemini Flash Image (дешёвый путь) |
| `RIVERFLOW_PRO_MODEL` | нет | `sourceful/riverflow-v2.5-pro` (выключен: `EDIT_ROUTE_PRO_ON_MESSY_HAIR=0`) |
| `EDIT_ROUTE_PRO_ON_MESSY_HAIR` | нет | `0` — всегда Gemini; `1` — Pro на messy hair + светлый фон |
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
