# Tochka payments — deploy checklist

Оплата привязана к `result_id`.

| Product | Price | Unlock |
|---------|-------|--------|
| Passport (`product=passport`) | 70 ₽ (`PRICE_KOPECKS`) | `digital.jpg` + `print.jpg` |
| Resume suit (`product=resume`) | 500 ₽ (`RESUME_PRICE_KOPECKS`) | `resume.jpg` |

Оплаты независимы: можно купить резюме без паспортного unlock и наоборот.

## Env (на API VPS `.env`)

| Variable | Prod |
|----------|------|
| `TOCHKA_ACCESS_TOKEN` | JWT Точки |
| `TOCHKA_CUSTOMER_CODE` | customerCode (или авто-resolve) |
| `TOCHKA_MERCHANT_ID` | optional |
| `TOCHKA_API_BASE_URL` | `https://enter.tochka.com` |
| `PUBLIC_BASE_URL` | `https://gosphoto.ru` |
| `PRICE_KOPECKS` | `7000` |
| `RESUME_PRICE_KOPECKS` | `50000` |
| `RESUME_UPSELL_ENABLED` | `1` (off → без suit-генерации) |
| `FREE_DOWNLOAD_UNLOCK` | `false` |
| `PAYMENT_SYNC_INTERVAL_SECONDS` | `30` |
| `PAYMENTS_DIR` | `/app/payments` (volume) |

## Webhook

- URL: `https://gosphoto.ru/api/payments/tochka/webhook`
- Nginx лендинга уже проксирует `/api/` → API
- В кабинете Точки: acquiring webhook → этот URL
- Подпись: JWT RS256 (публичный ключ Точки в коде)
- Webhook unlock ветвит по `product` в payment record (`passport` / `resume`)

## Smoke passport

1. `POST /api/process` → `result_id`
2. `GET /api/result/{id}` → `paid: false`, preview URLs
3. `GET .../digital.jpg` → 403
4. `POST /api/result/{id}/pay` → `payment_url`
5. Оплатить / webhook → `paid: true`
6. `GET .../digital.jpg` → 200

## Smoke resume upsell

1. Селфи с торсом → `POST /api/process` → `resume_offer: true` (ждёт оба edit)
2. `GET .../preview_resume.jpg` → 200 (watermark)
3. `GET .../resume.jpg` → 403
4. `POST /api/result/{id}/pay-resume` → `payment_url` (500 ₽)
5. Webhook → `paid_resume: true`
6. `GET .../resume.jpg` → 200

## Local without Tochka

- Без `TOCHKA_ACCESS_TOKEN` — stub client (`payment_url` fake)
- `FREE_DOWNLOAD_UNLOCK=true` — `POST .../pay` и `.../pay-resume` сразу unlock
