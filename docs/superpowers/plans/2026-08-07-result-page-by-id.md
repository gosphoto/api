# Result page by id — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After `/api/process`, store result by id and open a dedicated `/result/<id>` HTML page (Visafoto-like).

**Architecture:** API writes `results/<id>/{digital.jpg,print.jpg,meta.json}` and exposes GET endpoints. Landing keeps progress dialog on `/`, then redirects. Nginx maps `/result/<32-hex>` → `result.html`.

**Tech Stack:** FastAPI, local disk volume, static HTML/CSS/JS, nginx.

**Spec:** `docs/superpowers/specs/2026-08-07-result-page-by-id-design.md`

## Global Constraints

- Id: exactly 32 lowercase hex chars (`secrets.token_hex(16)`)
- No payment / watermark / email in this plan
- Landing must not depend on base64 for the result page (use API image URLs)
- Tenant/org N/A (gosphoto public)

## File map

| File | Role |
|------|------|
| `gosphoto-api/app/config.py` | `RESULTS_DIR`, `RESULTS_ENABLED` |
| `gosphoto-api/app/results.py` | save + load helpers |
| `gosphoto-api/app/main.py` | process returns `result_id`; GET result routes |
| `gosphoto-api/tests/test_results.py` | save/load + HTTP tests |
| `gosphoto-api/docker-compose*.yml` / deploy | mount `results` volume if needed |
| `gosphoto-landing/result.html` | dedicated result page |
| `gosphoto-landing/index.html` | redirect on success; remove `#result-screen` |
| `gosphoto-landing/css/styles.css` | keep result styles; tweak for full page |
| `gosphoto-landing/deploy/*.nginx.conf` | pretty `/result/<id>` |
| `.github/workflows/deploy.yml` (landing) | ensure nginx conf applied if applicable |

---

### Task 1: API — persist + GET result

**Repo:** gosphoto-api

- [ ] Add `RESULTS_DIR` / `RESULTS_ENABLED` in `app/config.py`
- [ ] Create `app/results.py`: `save_result(...)`, `load_meta(id)`, `path_for(id, name)` with id validation
- [ ] In `POST /api/process` after encode: call `save_result`, include `result_id` + `result_path` in JSON
- [ ] Add `GET /api/result/{id}`, `.../digital.jpg`, `.../print.jpg`
- [ ] Write tests in `tests/test_results.py` (tmpdir via env/monkeypatch)
- [ ] Run `pytest tests/test_results.py` — pass
- [ ] Bump health version note if needed (e.g. 0.8.0)
- [ ] Commit: `Store process results by id for /result/<id> pages.`

### Task 2: Landing — `result.html` + nginx

**Repo:** gosphoto-landing

- [ ] Add `result.html` with header, result products, specs table, actions (copy markup from current `#result-screen`)
- [ ] Script: parse id from path, fetch `/api/result/{id}`, bind UI, downloads via API URLs
- [ ] Nginx: `location ~ ^/result/[a-f0-9]{32}$ { try_files /result.html =404; }` in both conf files
- [ ] Commit: `Add dedicated result.html served at /result/<id>.`

### Task 3: Landing — redirect from process; remove inline result

**Repo:** gosphoto-landing

- [ ] On process success: require `data.result_id`, then `location.assign(/result/${id}?t=…)`
- [ ] Remove `#result-screen` markup and related show/hide/render helpers from `index.html`
- [ ] Keep progress dialog + error handling on `/`
- [ ] Commit: `Redirect to /result/<id> after process; drop inline result screen.`

### Task 4: Deploy + smoke

- [ ] Push api `main` → watch Deploy success
- [ ] Push landing `main` → watch Deploy success; apply nginx if workflow doesn't auto-reload
- [ ] Smoke: upload selfie → URL `/result/<hex>?t=…` → reload → downloads work
- [ ] Report PASS/FAIL with SHA + CI URLs

## Done when

Prod: process → `/result/<id>?t=…` full page; F5 works; digital + print + table visible.
