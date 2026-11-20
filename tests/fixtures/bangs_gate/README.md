# bangs_gate fixtures

Gate must reject uploads where bangs (чёлка) cover the eyebrows/eyes.

BiSeNet overlays for TDD: `overlays/{positive,negative}/{stem}.mask.png` +
`.overlay.png`. Regenerate with `scripts/bangs_gate_bisenet_overlays.py`.

| Split | Count | Expect |
|-------|------:|--------|
| `positive/` | ≥30 (includes `semenova.jpg`) | `reason=bangs_on_brows` |
| `negative/` | ≥30 from prod / clean studio | `reason != bangs_on_brows` |

Positives are resized `pairs/*/in.*` samples with brow-covering fringe.
Negatives are passport previews without fringe on the brows (some may fail
other gate checks, e.g. grayscale — that is OK). Light tip-touch on brows
must pass (not `bangs_on_brows`).
