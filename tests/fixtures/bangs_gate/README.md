# bangs_gate fixtures

Gate must reject uploads where bangs (чёлка) cover the eyebrows.

| Split | Count | Expect |
|-------|-------|--------|
| `positive/` | ≥30 (includes `semenova.jpg`) | `reason=bangs_on_brows` |
| `negative/` | ≥30 from prod `results/*/preview_digital.jpg` | `reason != bangs_on_brows` |

Positives are resized `pairs/*/in.*` samples with brow-covering fringe.
Negatives are passport previews without fringe on the brows (some may fail other gate checks, e.g. grayscale — that is OK).
