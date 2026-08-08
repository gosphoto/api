# Gosphoto API

Бэкенд для https://gosphoto.ru

## Live API (сайт)

| Метод | Путь | Назначение |
|-------|------|------------|
| `GET` | `/health` | liveness / config |
| `POST` | `/api/process/nano-banana` | gate → Nano Banana white-bg → 35×45 + лист 10×15 |
| `GET` | `/api/result/{id}` | meta + URL цифрового и печатного JPEG |
| `GET` | `/api/result/{id}/digital.jpg` | 35×45 |
| `GET` | `/api/result/{id}/print.jpg` | 10×15 (4 копии) |

Удалены (сайт не вызывал): `/api/validate`, `/api/edit`, `/api/crop`, `/api/process` (gpt-image).

JSON process: `ok`, `result_id`, `compliance`, `print_sheet`, …

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

Production API: `80.87.196.33` (см. `docs/` / deploy workflow).  
Nginx на лендинге проксирует `/api/` и `/health`.

Push в `main` → GitHub Actions deploy (если настроено).

## Secrets

https://github.com/gosphoto/api/settings/secrets/actions

| Secret | Обязателен | Назначение |
|--------|------------|------------|
| `DEPLOY_SSH_PRIVATE_KEY` | да | SSH на VPS |
| `DEPLOY_USER` | да | SSH user |
| `OPENROUTER_API_KEY` | да | Nano Banana / image edit |
| `NANO_BANANA_MODEL` | нет | default `google/gemini-3-pro-image-preview` |

## Local

```bash
cp .env.example .env
docker compose up --build
curl http://127.0.0.1:8091/health
curl -F file=@selfie.jpg "http://127.0.0.1:8091/api/process/nano-banana"
```
