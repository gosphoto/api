# Result page by id (Visafoto-like) — Design

**Date:** 2026-08-07  
**Repos:** gosphoto-api + gosphoto-landing  
**Status:** approved

## Problem

Result UI lives as a hidden section on the landing (`#result-screen`). After process, the user stays on `/` with no shareable/reloadable URL. Visafoto uses `/result/<id>?t=…` — a dedicated page that survives refresh.

## Goal

After successful photo process, navigate to a **dedicated result screen** at its own path with its own HTML. Result data is stored server-side and loaded by id.

## Non-goals

- Payment, watermarks, email delivery
- Redis/S3 (v1 = local disk volume)
- TTL cleanup job (note as follow-up)
- Changing passport crop / compliance rules

## URL

```
https://gosphoto.ru/result/<32-hex>?t=<unix_ms>
```

- `<id>` — opaque hex token (16 bytes → 32 hex chars), like Visafoto order id
- `t` — client cache-buster only; ignored by server

Direct open of `/result` without id → redirect to `/`.

## API (gosphoto-api)

### Persist on process

On successful `POST /api/process` (`format=json`):

1. Generate `result_id = secrets.token_hex(16)`
2. Write under `RESULTS_DIR/<result_id>/`:
   - `digital.jpg` — 35×45 passport JPEG
   - `print.jpg` — 10×15 sheet JPEG
   - `meta.json` — compliance, dimensions, dpi, print meta, timestamps (no base64)
3. JSON response adds:
   - `result_id`
   - `result_path`: `/result/<result_id>`
4. Keep existing `image_base64` / `print_sheet.image_base64` for one release (compat); landing will stop relying on them for the result page.

Config: `RESULTS_DIR` (default `…/results`), `RESULTS_ENABLED` (default on). Mount volume in compose like `pairs/`.

### Read

- `GET /api/result/{id}` → JSON:
  - `ok`, `result_id`
  - `digital_url`, `print_url` (e.g. `/api/result/{id}/digital.jpg`)
  - passport width/height/dpi, compliance, print_sheet meta (no base64)
  - 404 if missing
- `GET /api/result/{id}/digital.jpg` → JPEG
- `GET /api/result/{id}/print.jpg` → JPEG

Validate id: `^[a-f0-9]{32}$` only (path traversal safe).

## Landing (gosphoto-landing)

### Flow

1. User uploads on `/` → progress dialog (`#result-dialog`) unchanged
2. On success: `location.assign(/result/${result_id}?t=${Date.now()})`
3. On gate/API fail: stay on `/`, error in dialog (no navigation)

### New page

- File: `result.html` (own document: head, chrome header/footer, result UI, script)
- Styles: reuse `css/styles.css`; result-only rules stay in that file or a small `css/result.css` if needed
- On load: parse id from `location.pathname` (`/result/<id>`), `GET /api/result/{id}`, render previews + table + download links pointing at API JPEG URLs
- Missing/invalid id or 404 → message + link «Сделать фото» → `/`
- «Сделать ещё фото» → `/`

Remove `#result-screen` (and its show/hide JS) from `index.html`.

### Nginx

Pretty URL without changing static root layout:

```nginx
location ~ ^/result/[a-f0-9]{32}$ {
    try_files /result.html =404;
}
```

Keep `/api/` proxy as today. Deploy workflow copies `result.html` with the rest of the site. Update `deploy/*.nginx.conf` and apply on VPS in the same deploy job if the workflow already syncs nginx; otherwise document a one-line remote include.

## Security / privacy

- Ids are unguessable (128-bit). Anyone with the link can view/download (same as Visafoto).
- Do not list directory indexes.
- Do not put PII in `meta.json` beyond image metrics.
- Optional later: TTL deletion of old `results/`.

## Success criteria

1. After upload+process on prod, browser URL is `/result/<32-hex>?t=…`
2. Full-page result (not overlay on landing hero)
3. Reload same URL still shows digital + print + compliance table
4. Downloads work via `/api/result/.../*.jpg`
5. CI green + deploy for api and landing

## Self-review

- No placeholders left for core flow
- Payment explicitly out of scope
- Compat base64 kept one release to avoid breaking any external caller
- Nginx regex matches only 32-hex paths
