# Tochka payments — deploy checklist

Оплата привязана к `result_id`. Одна оплата (100 ₽) открывает `digital.jpg` + `print.jpg`.

## Env (на API VPS `.env`)

| Variable | Prod |
|----------|------|
| `TOCHKA_ACCESS_TOKEN` | JWT Точки |
| `TOCHKA_CUSTOMER_CODE` | customerCode (или авто-resolve) |
| `TOCHKA_MERCHANT_ID` | optional |
| `TOCHKA_API_BASE_URL` | `https://enter.tochka.com` |
| `PUBLIC_BASE_URL` | `https://gosphoto.ru` |
| `PRICE_KOPECKS` | `10000` |
| `FREE_DOWNLOAD_UNLOCK` | `false` |
| `PAYMENT_SYNC_INTERVAL_SECONDS` | `30` |
| `PAYMENTS_DIR` | `/app/payments` (volume) |

## Webhook

- URL: `https://gosphoto.ru/api/payments/tochka/webhook`
- Nginx лендинга уже проксирует `/api/` → API
- В кабинете Точки: acquiring webhook → этот URL
- Подпись: JWT RS256 (публичный ключ Точки в коде)

## Smoke

1. `POST /api/process` → `result_id`
2. `GET /api/result/{id}` → `paid: false`, preview URLs
3. `GET .../digital.jpg` → 403
4. `POST /api/result/{id}/pay` → `payment_url`
5. Оплатить / webhook → `paid: true`
6. `GET .../digital.jpg` → 200

## Local without Tochka

- Без `TOCHKA_ACCESS_TOKEN` — stub client (`payment_url` fake)
- `FREE_DOWNLOAD_UNLOCK=true` — `POST .../pay` сразу unlock
