# OpenRouter Clean-BG + Crop Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove coloured fringe on passport photos by using OpenRouter `gpt-image-1` (transparent PNG → local white composite), then keep local crop geometry + compliance retries.

**Architecture:** Edit stage calls OpenRouter Image API with `background: transparent` when configured; decode RGBA and composite onto `#FFFFFF`; light edge defringe + existing `force_white_background`; crop stage unchanged except continued post-crop whitening. Local MediaPipe/rembg remains fallback.

**Tech Stack:** Python 3.11, FastAPI, OpenCV, NumPy, httpx, OpenRouter `/api/v1/images`, pytest (new `tests/`).

**Spec:** `docs/superpowers/specs/2026-08-06-openrouter-clean-bg-crop-design.md`

## Global Constraints

- Primary model: `openai/gpt-image-1`; keep Gemini usable via `OPENROUTER_IMAGE_MODEL` env.
- Prefer `EDIT_BACKEND=openrouter` when `OPENROUTER_API_KEY` is set; without key, local cutout must still work.
- Do not change crop geometry search space / landmarker indices in v1.
- No fal.ai / remove.bg in v1.
- No beauty/identity-altering prompts.
- Identity-safe: only background replacement.

## File map

| File | Responsibility |
|------|----------------|
| `app/config.py` | Defaults: model, optional `OPENROUTER_TRANSPARENT_BG` |
| `app/compose_bg.py` | Pure alpha→white composite (new, testable without MediaPipe) |
| `app/openrouter.py` | Prompt, transparent/png request, return raw bytes + mime hint |
| `app/whitening.py` | Edge defringe band + existing force white |
| `app/edit.py` | OR-first wiring, decode/composite, local fallback |
| `docker-compose.yml` | Default model + backend when key present |
| `.github/workflows/deploy.yml` | Write `EDIT_BACKEND=openrouter` when key secret set |
| `tests/test_compose_bg.py` | Unit tests for composite |
| `tests/test_openrouter_payload.py` | Unit tests for request payload builder |
| `tests/test_defringe.py` | Unit tests for edge defringe on synthetic images |

---

### Task 1: Alpha→white composite helper

**Files:**
- Create: `app/compose_bg.py`
- Create: `tests/test_compose_bg.py`
- Test: `tests/test_compose_bg.py`

**Interfaces:**
- Consumes: OpenCV/`numpy` BGR or BGRA arrays
- Produces: `composite_on_white(bgr_or_bgra: np.ndarray) -> np.ndarray` — always 3-channel BGR uint8 on white

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compose_bg.py
import numpy as np
import cv2
from app.compose_bg import composite_on_white


def test_opaque_bgr_unchanged_shape():
    img = np.full((10, 10, 3), (40, 80, 120), np.uint8)
    out = composite_on_white(img)
    assert out.shape == (10, 10, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_fully_transparent_becomes_white():
    bgra = np.zeros((8, 8, 4), np.uint8)
    bgra[:, :, :3] = (0, 0, 255)  # red in BGR sense if mis-ordered; use B=0,G=0,R=255
    bgra[:, :, 0] = 0
    bgra[:, :, 1] = 0
    bgra[:, :, 2] = 255
    bgra[:, :, 3] = 0
    out = composite_on_white(bgra)
    assert np.all(out == 255)


def test_half_alpha_blends_toward_white():
    bgra = np.zeros((4, 4, 4), np.uint8)
    bgra[:, :, 0] = 0
    bgra[:, :, 1] = 0
    bgra[:, :, 2] = 0
    bgra[:, :, 3] = 128
    out = composite_on_white(bgra)
    # ~127–128 gray toward white
    assert 120 <= int(out[0, 0, 0]) <= 140
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/antonbutov/Documents/MYPROJECTS/gosphoto-api && .venv311/bin/python -m pytest tests/test_compose_bg.py -v`  
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `app.compose_bg`

- [ ] **Step 3: Implement `app/compose_bg.py`**

```python
"""Composite cutout images onto solid #FFFFFF."""

from __future__ import annotations

import cv2
import numpy as np


def composite_on_white(bgr_or_bgra: np.ndarray) -> np.ndarray:
    """Return BGR uint8 on white. Opaque 3-channel images pass through."""
    if bgr_or_bgra.ndim != 3:
        raise ValueError("expected HWC image")
    h, w, c = bgr_or_bgra.shape
    if c == 3:
        return bgr_or_bgra.copy()
    if c != 4:
        raise ValueError(f"expected 3 or 4 channels, got {c}")

    # OpenCV IMREAD_UNCHANGED: BGRA
    bgr = bgr_or_bgra[:, :, :3].astype(np.float32)
    a = bgr_or_bgra[:, :, 3].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    out = bgr * a[:, :, None] + white * (1.0 - a[:, :, None])
    return np.clip(out, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv311/bin/python -m pytest tests/test_compose_bg.py -v`  
Expected: PASS (install pytest if missing: `.venv311/bin/pip install pytest`)

- [ ] **Step 5: Commit**

```bash
git add app/compose_bg.py tests/test_compose_bg.py
git commit -m "$(cat <<'EOF'
Add alpha-to-white composite helper for cutouts.

EOF
)"
```

---

### Task 2: Config defaults + OpenRouter transparent payload

**Files:**
- Modify: `app/config.py`
- Modify: `app/openrouter.py`
- Create: `tests/test_openrouter_payload.py`

**Interfaces:**
- Consumes: `config.OPENROUTER_*`, image bytes
- Produces:
  - `build_edit_payload(image_bytes: bytes, mime: str) -> dict` (pure, testable)
  - `edit_selfie(...)` still returns `bytes` of image (PNG or JPEG)
  - `EDIT_PROMPT` updated for anti-fringe

- [ ] **Step 1: Write failing payload tests**

```python
# tests/test_openrouter_payload.py
from app.openrouter import build_edit_payload, EDIT_PROMPT


def test_prompt_mentions_no_fringe():
    lower = EDIT_PROMPT.lower()
    assert "fringe" in lower or "spill" in lower or "halo" in lower


def test_payload_uses_transparent_png_when_enabled(monkeypatch):
    monkeypatch.setattr("app.openrouter.config.OPENROUTER_IMAGE_MODEL", "openai/gpt-image-1")
    monkeypatch.setattr("app.openrouter.config.OPENROUTER_TRANSPARENT_BG", True)
    payload = build_edit_payload(b"fake", "image/jpeg")
    assert payload["model"] == "openai/gpt-image-1"
    assert payload["output_format"] == "png"
    assert payload["background"] == "transparent"
    assert payload["aspect_ratio"] == "3:4"
    assert payload["input_references"]


def test_payload_omits_transparent_when_disabled(monkeypatch):
    monkeypatch.setattr("app.openrouter.config.OPENROUTER_TRANSPARENT_BG", False)
    payload = build_edit_payload(b"fake", "image/jpeg")
    assert "background" not in payload
    assert payload["output_format"] == "jpeg"
```

- [ ] **Step 2: Run tests — expect fail**

Run: `.venv311/bin/python -m pytest tests/test_openrouter_payload.py -v`  
Expected: FAIL (`build_edit_payload` missing / prompt without fringe)

- [ ] **Step 3: Update `app/config.py`**

Add / change:

```python
OPENROUTER_IMAGE_MODEL = os.getenv(
    "OPENROUTER_IMAGE_MODEL", "openai/gpt-image-1"
)
# When true, request transparent PNG then composite locally onto white.
OPENROUTER_TRANSPARENT_BG = os.getenv(
    "OPENROUTER_TRANSPARENT_BG", "1"
).strip().lower() in ("1", "true", "yes", "on")
```

Keep existing `EDIT_BACKEND` default `local` in code; deploy/compose will set `openrouter` when key exists (Task 5). Document in comment that production with a key should use `EDIT_BACKEND=openrouter`.

- [ ] **Step 4: Refactor `app/openrouter.py`**

Replace `EDIT_PROMPT` with:

```python
EDIT_PROMPT = (
    "Replace ONLY the background with a transparent alpha channel "
    "(or solid pure white #FFFFFF if transparency is unavailable). "
    "Do not change the person's face, skin, hair, eyes, mouth, clothes, or identity. "
    "No beauty filter, no morphing, no retouching, no makeup. "
    "Keep original sharpness. Remove any coloured fringe, color spill, or halo "
    "along hair and shoulders. Studio ID / Russian passport style, "
    "frontal head-and-shoulders."
)
```

Add:

```python
def build_edit_payload(image_bytes: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    transparent = bool(config.OPENROUTER_TRANSPARENT_BG)
    payload: dict[str, Any] = {
        "model": config.OPENROUTER_IMAGE_MODEL,
        "prompt": EDIT_PROMPT,
        "aspect_ratio": "3:4",
        "output_format": "png" if transparent else "jpeg",
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        ],
    }
    if transparent:
        payload["background"] = "transparent"
    return payload
```

Change `edit_selfie` to `payload = build_edit_payload(image_bytes, mime)` instead of inline dict. Keep response parsing (b64_json / url).

- [ ] **Step 5: Run tests — expect pass**

Run: `.venv311/bin/python -m pytest tests/test_openrouter_payload.py tests/test_compose_bg.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/openrouter.py tests/test_openrouter_payload.py
git commit -m "$(cat <<'EOF'
Request transparent PNG from OpenRouter for cleaner cutouts.

EOF
)"
```

---

### Task 3: Edge defringe in whitening

**Files:**
- Modify: `app/whitening.py`
- Create: `tests/test_defringe.py`

**Interfaces:**
- Consumes: BGR image
- Produces: `defringe_background(bgr: np.ndarray, band: int = 3) -> np.ndarray`  
  Logic without MediaPipe: pixels near near-white background that still have chroma get pulled to white. Then `force_white_background` may call it at the end, or `edit.py` calls it after composite.

Prefer a **MediaPipe-free** defringe for unit tests:

```python
def defringe_near_white(bgr: np.ndarray, *, luma_min: int = 200, chroma_max: float = 18.0) -> np.ndarray:
    """Force near-white but tinted pixels to pure white (fringe cleanup)."""
```

- [ ] **Step 1: Write failing test**

```python
# tests/test_defringe.py
import numpy as np
from app.whitening import defringe_near_white


def test_tinted_near_white_becomes_white():
    # Light pink-ish near-white (spill)
    img = np.full((20, 20, 3), (235, 240, 255), np.uint8)  # BGR: high R channel tint
    out = defringe_near_white(img)
    assert np.all(out >= 250)


def test_dark_subject_preserved():
    img = np.full((20, 20, 3), (30, 40, 50), np.uint8)
    out = defringe_near_white(img)
    np.testing.assert_array_equal(out, img)
```

- [ ] **Step 2: Run — expect fail**

Run: `.venv311/bin/python -m pytest tests/test_defringe.py -v`  
Expected: FAIL (`defringe_near_white` missing)

- [ ] **Step 3: Implement in `app/whitening.py`**

```python
def defringe_near_white(
    bgr: np.ndarray,
    *,
    luma_min: int = 200,
    chroma_max: float = 18.0,
) -> np.ndarray:
    """Bleach near-white tinted pixels (classic cutout fringe) to #FFFFFF."""
    if bgr.size == 0:
        return bgr
    out = bgr.copy()
    f = bgr.astype(np.float32)
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # chroma vs gray: distance from equal RGB
    mean_c = f.mean(axis=2, keepdims=True)
    chroma = np.linalg.norm(f - mean_c, axis=2)
    mask = (luma >= float(luma_min)) & (chroma >= 1.0) & (chroma <= float(chroma_max) * 3)
    # also catch mild tints: high luma and not already pure white
    near = (luma >= float(luma_min)) & (np.min(f, axis=2) < 250)
    mask = mask | near
    out[mask] = 255
    return out
```

At the **end** of `force_white_background`, before return:

```python
out = defringe_near_white(np.clip(out, 0, 255).astype(np.uint8))
out[subject > 0] = bgr[subject > 0]  # re-protect subject after defringe
return out
```

Adjust order carefully: apply defringe only where `subject == 0` (and corner chips), never overwrite protected subject.

```python
cleaned = defringe_near_white(np.clip(out, 0, 255).astype(np.uint8))
protect = subject > 0
cleaned[protect] = bgr[protect]
return cleaned
```

- [ ] **Step 4: Run tests — expect pass**

Run: `.venv311/bin/python -m pytest tests/test_defringe.py -v`  
Expected: PASS (tune thresholds if pink sample fails)

- [ ] **Step 5: Commit**

```bash
git add app/whitening.py tests/test_defringe.py
git commit -m "$(cat <<'EOF'
Defringe near-white tinted background pixels after whitening.

EOF
)"
```

---

### Task 4: Wire edit stage (OR-first + composite)

**Files:**
- Modify: `app/edit.py`
- Create: `tests/test_edit_or_path.py` (mocked httpx / `edit_selfie`)

**Interfaces:**
- Consumes: `edit_selfie`, `composite_on_white`, `force_white_background`, `white_background_local`
- Produces: `run_edit_stage` behaviour:
  1. If `EDIT_BACKEND in (openrouter, auto)` and key → try OR first
  2. Decode bytes with `cv2.IMREAD_UNCHANGED` → `composite_on_white` → `force_white`
  3. On OR failure with `auto`/`openrouter` → local path (openrouter-only still falls back to local with warning, matching robustness goal)

Spec: OR fail → local. So even `EDIT_BACKEND=openrouter` should fall back to local after logging (better UX than 502 when OR flakes).

- [ ] **Step 1: Write failing test with mocks**

```python
# tests/test_edit_or_path.py
import numpy as np
import cv2
from unittest.mock import patch
from app import edit as edit_mod


def _png_rgba_red_on_transparent() -> bytes:
    bgra = np.zeros((16, 16, 4), np.uint8)
    bgra[4:12, 4:12, 2] = 255  # R
    bgra[4:12, 4:12, 3] = 255
    ok, buf = cv2.imencode(".png", bgra)
    assert ok
    return buf.tobytes()


def test_or_path_composites_alpha(monkeypatch):
    monkeypatch.setattr(edit_mod.config, "EDIT_BACKEND", "openrouter")
    monkeypatch.setattr(edit_mod.config, "OPENROUTER_API_KEY", "test-key")

    def fake_edit(data, mime="image/jpeg"):
        return _png_rgba_red_on_transparent()

    with patch.object(edit_mod, "edit_selfie", side_effect=fake_edit):
        # skip force_white MediaPipe by stubbing
        with patch.object(edit_mod, "force_white_background", side_effect=lambda im, tol=52: im):
            out, meta = edit_mod.run_edit_stage(b"jpeg-bytes", "image/jpeg")
    assert meta.get("cutout") == "openrouter"
    assert out.shape[2] == 3
    # corners should be white after composite
    assert np.all(out[0, 0] == 255)
```

- [ ] **Step 2: Run — expect fail / wrong order**

Run: `.venv311/bin/python -m pytest tests/test_edit_or_path.py -v`  
Expected: FAIL until edit order is OpenRouter-first

- [ ] **Step 3: Rewrite `run_edit_stage` priority**

Replace strategy comment and body so that when key present and backend is `openrouter` or `auto`, **try OpenRouter first**, then local:

```python
def _decode_any(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def run_edit_stage(data: bytes, mime: str = "image/jpeg") -> tuple[np.ndarray, dict[str, Any]]:
    backend = config.EDIT_BACKEND
    use_or = backend in ("openrouter", "auto") and bool(config.OPENROUTER_API_KEY)
    use_local = backend in ("local", "auto", "openrouter", "")
    meta: dict[str, Any] = {"stage": "edit"}
    or_err: Exception | None = None

    if use_or:
        try:
            from .compose_bg import composite_on_white

            raw = edit_selfie(data, mime=mime)
            decoded = _decode_any(raw)
            if decoded is None:
                raise RuntimeError("Edited image decode failed")
            edited = composite_on_white(decoded)
            edited = force_white_background(edited, tol=52)
            meta.update(
                {
                    "model": config.OPENROUTER_IMAGE_MODEL,
                    "cutout": "openrouter",
                    "width": int(edited.shape[1]),
                    "height": int(edited.shape[0]),
                }
            )
            return edited, meta
        except Exception as e:
            or_err = e
            log.warning("OpenRouter edit failed, falling back to local: %s", e)
            if backend == "openrouter" and not use_local:
                raise

    if use_local:
        try:
            src = _decode_image(data)
            if src is None:
                raise RuntimeError("decode_error")
            edited, local_meta = edit_selfie_local(src)
            meta.update(local_meta)
            meta["model"] = local_meta.get("cutout", "mediapipe")
            if or_err is not None:
                meta["openrouter_fallback"] = str(or_err)[:200]
            return edited, meta
        except Exception:
            if or_err:
                raise or_err
            raise

    if or_err:
        raise or_err
    raise RuntimeError(f"No edit backend available (EDIT_BACKEND={backend})")
```

Import `composite_on_white` at top of file if preferred.

- [ ] **Step 4: Run unit tests**

Run: `.venv311/bin/python -m pytest tests/ -v`  
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add app/edit.py tests/test_edit_or_path.py
git commit -m "$(cat <<'EOF'
Prefer OpenRouter edit with alpha composite and local fallback.

EOF
)"
```

---

### Task 5: Deploy / compose defaults

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.github/workflows/deploy.yml`

**Interfaces:**
- Env: `OPENROUTER_IMAGE_MODEL` default `openai/gpt-image-1`
- `EDIT_BACKEND=openrouter` when deploying with key; keep local-only VPS safe if key empty

- [ ] **Step 1: Update `docker-compose.yml` environment**

```yaml
EDIT_BACKEND: ${EDIT_BACKEND:-openrouter}
EDIT_CUTOUT: ${EDIT_CUTOUT:-mediapipe}
OPENROUTER_IMAGE_MODEL: ${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-1}
OPENROUTER_TRANSPARENT_BG: ${OPENROUTER_TRANSPARENT_BG:-1}
OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:-}
```

Note in comment: if key empty, `edit.py` still falls through to local when backend is openrouter (after OR failure) — OR will fail fast without key. Safer default for empty key:

In `run_edit_stage`, `use_or` already requires key. So `EDIT_BACKEND=openrouter` without key → skip OR → local. Ensure that path works (Task 4 `use_local` includes `openrouter`).

- [ ] **Step 2: Update deploy workflow `.env` writer**

Change the block that currently forces `EDIT_BACKEND=local` to:

```bash
MODEL="${OPENROUTER_IMAGE_MODEL:-openai/gpt-image-1}"
if [ -n "${OPENROUTER_API_KEY}" ]; then
  EDIT_BACKEND_VAL=openrouter
else
  EDIT_BACKEND_VAL=local
fi
{
  echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}"
  echo "OPENROUTER_IMAGE_MODEL=${MODEL}"
  echo "OPENROUTER_TRANSPARENT_BG=1"
  echo "EDIT_BACKEND=${EDIT_BACKEND_VAL}"
  echo "EDIT_CUTOUT=mediapipe"
} | ssh ...
```

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml .github/workflows/deploy.yml
git commit -m "$(cat <<'EOF'
Default deploy to OpenRouter gpt-image-1 when API key is set.

EOF
)"
```

---

### Task 6: Manual smoke (no live OR required for CI)

**Files:** none required (checklist)

- [ ] **Step 1: Local unit suite**

Run: `.venv311/bin/python -m pytest tests/ -v`  
Expected: PASS

- [ ] **Step 2: Health + local edit without key**

```bash
# with EDIT_BACKEND=local or openrouter without key
curl -fsS http://127.0.0.1:8091/health
```

- [ ] **Step 3: Optional live OR** (only if `OPENROUTER_API_KEY` available)

```bash
curl -fsS -F "file=@sample.jpg" "http://127.0.0.1:8091/api/process?format=json" | head
# Inspect returned image corners: should be near-white; no coloured halo
```

- [ ] **Step 4: Final commit if any smoke fixes**

Only if code changes were needed during smoke.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| gpt-image-1 default | 2, 5 |
| transparent PNG → local white | 1, 2, 4 |
| anti-fringe prompt | 2 |
| force_white safety net | 4 (+ existing crop) |
| edge defringe | 3 |
| OR fail → local | 4 |
| crop geometry unchanged | (no task — intentional) |
| deploy/compose defaults | 5 |
| no fal.ai v1 | (omitted) |

## Self-review notes

- No TBD placeholders.
- `composite_on_white` / `build_edit_payload` / `defringe_near_white` names consistent across tasks.
- Crop.py intentionally untouched beyond existing whitening call.
