# Resume suit upsell

> Experimental · 2026-08-12

После успешного gate: MediaPipe Pose проверяет видимость торса. Если ок и
`RESUME_UPSELL_ENABLED=1`, параллельно с паспортным Riverflow генерируется
фото в деловом костюме (лёгкая ретушь). На `/result/{id}` — downscale-превью (без watermark)
и отдельная оплата Точка **300 ₽** (`POST /api/result/{id}/pay-resume`).

## Flags

| Env | Default | Role |
|-----|---------|------|
| `RESUME_UPSELL_ENABLED` | `1` | Master switch |
| `RESUME_PRICE_KOPECKS` | `30000` | 300 ₽ |
| `POSE_MODEL_PATH` | `models/pose_landmarker_lite.task` | Pose model |
| `TORSO_MIN_VISIBILITY` | `0.45` | Shoulder visibility |
| `TORSO_MIN_SHOULDER_DROP` | `0.06` | Nose→shoulder drop |
| `TORSO_MIN_SHOULDER_WIDTH` | `0.12` | Shoulder span |

## Failure modes

- Нет торса / Pose model missing → только паспорт, без апселла
- Suit edit error → паспорт сохраняется, `resume_offer=false`
- `RESUME_UPSELL_ENABLED=0` → поведение как до фичи
