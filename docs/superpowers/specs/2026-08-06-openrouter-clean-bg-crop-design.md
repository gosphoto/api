# Design: clean white background + passport crop quality

**Date:** 2026-08-06  
**Status:** approved (design dialogue)  
**Problem:** After edit/crop, coloured fringe (color spill) remains around hair/shoulders; final 35×45 quality needs both clean contour and RF geometry.

## Goal

Deliver a Gosuslugi-ready 35×45 @ 300 dpi photo with:

1. **Clean contour** — solid `#FFFFFF` background, no coloured halo at hair/shoulders/ears.
2. **Geometry** — face ~70–80% of height, top margin ~5 mm (~0.09–0.14), roll-corrected; keep existing compliance retries.

## Non-goals (v1)

- fal.ai BRIA / remove.bg integration (interface only if easy; not required to ship).
- Changing MediaPipe landmarker / crop geometry formulas beyond current retries.
- New HTTP endpoints.
- Beauty filters or identity-altering retouch.

## Pipeline

```
selfie → gate (local)
      → edit (OpenRouter remote: transparent/white bg, no fringe)
      → composite alpha → #FFFFFF (local)
      → force_white_background (local safety net)
      → crop (local: roll + 35×45 + compliance retries)
      → force_white_background on each crop attempt
      → JPEG 300 dpi
```

Fallback: if OpenRouter fails → local MediaPipe/rembg cutout (current behaviour), then same whitening/crop path.

## Approach (chosen)

Remote matting/edit via OpenRouter, then local white composite + crop compliance.

Rejected for v1 as primary:

- **Local-only whitening tweaks** — cannot remove strong colour spill from soft matte.
- **Cost hybrid (remote only when fringe detected)** — deferred; quality first.

## OpenRouter edit

### Model

| Role | Model | Notes |
|------|--------|--------|
| Primary | `openai/gpt-image-1` | Supports `input_references`, `background: transparent`, quality tiers |
| Fallback env default | `google/gemini-2.5-flash-image` | Cheaper; no transparent bg param — use opaque white via prompt |

Config: `OPENROUTER_IMAGE_MODEL` default → `openai/gpt-image-1`.  
`EDIT_BACKEND`: prefer `openrouter` when `OPENROUTER_API_KEY` is set (compose/deploy); keep `local` / `auto` behaviour for hosts without a key.

### Request

- Endpoint: existing `POST /api/v1/images` via `app/openrouter.py`.
- `input_references`: selfie data URL.
- `output_format`: `png` when using transparent background.
- `background`: `transparent` when the endpoint supports it; otherwise omit and ask for solid `#FFFFFF` in the prompt.
- `aspect_ratio`: `3:4` (unchanged).
- Prompt must require:
  - Replace **only** background.
  - No face/hair/clothes/identity change; no beauty/morph.
  - No coloured fringe / color spill along silhouette.
  - Studio passport / head-and-shoulders framing preserved (crop still done locally).

### Local compose

If response is PNG with alpha:

1. Decode BGRA/RGBA.
2. `out = rgb * a + 255 * (1 - a)`.
3. Optional light edge defringe in `whitening.py` (dilate fringe band outside protected subject; blend toward white where chroma/spill detected).
4. `force_white_background` as today.

If response is opaque JPEG/PNG without alpha: skip compose; still run `force_white_background`.

## Crop stage

Keep `run_crop_stage` geometry attempts and `_score_compliance`.  
After each attempt: `force_white_background(cropped, tol=…)`.  
No change to crown/face/top_margin search space in v1 unless compliance regressions appear after cleaner edit inputs.

## Code touch list

| File | Change |
|------|--------|
| `app/config.py` | Default image model; env knobs for transparent/png if needed |
| `app/openrouter.py` | Prompt, transparent/png, alpha-capable decode path |
| `app/edit.py` | OpenRouter-first when configured; alpha→white; local fallback |
| `app/whitening.py` | Light edge defringe without bleaching face/hair core |
| `app/crop.py` | Unchanged geometry; retain post-crop whitening |
| `docker-compose.yml` / deploy | `EDIT_BACKEND=openrouter` when key present; new model default |

## Error handling

- OpenRouter 4xx / timeout / empty image → log warning → local cutout.
- Local cutout also fails → raise → API 502 (existing).
- Crop failures / empty crop → existing retry / center fallback.

## Success criteria

- On selfies with coloured walls: no visible coloured halo on final crop.
- Corner chips meet `bg_white_ok` (mean ≥ 245 per channel).
- Compliance `face_ratio_ok` / `top_margin_ok` / `size_ok` still reachable via existing retries.
- Without `OPENROUTER_API_KEY`, pipeline still works on local cutout.

## Test plan

1. `/api/edit` with coloured-background selfie + OR key → inspect edges.
2. `/api/crop` on that edit → 413×531, compliance JSON.
3. `/api/process` end-to-end.
4. Kill OR / invalid key → confirm local fallback still returns an image.
5. Spot-check identity: face not beautified vs input.
