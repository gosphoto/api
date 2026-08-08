# Riverflow v2.5 Fast vs Pro (gosphoto edit)

> Проверено: 2026-08-08 · источники: prod smoke на 80.87.196.33 + визуальная оценка оператора · OpenRouter model cards

## Вердикт

**Живой edit остаётся на `sourceful/riverflow-v2.5-pro`.**  
`sourceful/riverflow-v2.5-fast` дал заметный выигрыш по latency, но качество выходных фото для Госуслуг / паспорта **неприемлемо** — откатили в тот же день.

**С чего начать:** не включать Fast на проде; если снова экспериментировать — только A/B на staging / одной паре, не default в deploy.

## Факты

| Факт | Источник | Статус |
|------|----------|--------|
| OpenRouter id Fast: `sourceful/riverflow-v2.5-fast` | [OpenRouter · Riverflow V2.5 Fast](https://openrouter.ai/sourceful/riverflow-v2.5-fast) | ✅ |
| OpenRouter id Pro: `sourceful/riverflow-v2.5-pro` | [OpenRouter · Sourceful](https://openrouter.ai/sourceful) | ✅ |
| Fast позиционируется как latency-optimized; Pro — quality / control | model cards выше | ✅ |
| Smoke Fast на том же селфи `i (8)` / pair `20260808T142432Z_i_8`: wall **~81 с**, `compliance.pass=true`, result `864670da0e5871634720aa2109f98836`, pair `20260808T143650Z_smoke-fast` | docker logs + meta на prod 2026-08-08 | ✅ |
| Тот же кадр на Pro ранее: wall **~129–159 с**, pass=true, result `e7b9342d0aa6ee08c5075034a29f0dd5` | gap health / access.log 2026-08-08 | ✅ |
| После просмотра live-результатов Fast оператор: «фотки очень некачественные» → откат на Pro | решение сессии 2026-08-08 | ✅ |
| Prod `.env` + health снова `riverflow_model=sourceful/riverflow-v2.5-pro` | curl `/health` после recreate | ✅ |
| Коммит Fast (`fc55760`) откатиться в default/deploy; deploy **force**-пишет `RIVERFLOW_MODEL` | `.github/workflows/deploy.yml` | ✅ |

## Сравнение (одна пара)

| | Pro | Fast |
|--|-----|------|
| Model | `sourceful/riverflow-v2.5-pro` | `sourceful/riverflow-v2.5-fast` |
| Reasoning (наш default) | `medium` | `medium` |
| Resolution | `1K` | `1K` |
| Latency (тот же input) | ~2.2–2.6 мин | ~81 с |
| Soft compliance | pass | pass |
| Качество для продукта | ок (оставлено) | **отклонено** оператором |

`compliance.pass` **не** достаточный критерий: gate/crop могут пройти при визуально слабом edit (края, текстура, «AI-гладкость»).

## Не брать / оговорки

- Не ставить Fast default ради UX прогресса (~2–3 мин на Pro) без отдельного quality bar.
- Не опираться только на `pass=true` / face_ratio при выборе модели.
- Повторный прогон Fast — только с явным A/B и сохранением pair IN/OUT; не менять prod без подтверждения.
- Ускорение без смены модели: пробовать `RIVERFLOW_REASONING=low` на **Pro** (отдельный эксперимент; на 2026-08-08 не гоняли).

## История правок

| Дата | Что изменили |
|------|----------------|
| 2026-08-08 | Включили Fast на prod (`fc55760`), smoke ~81 с; откат на Pro по качеству; эта заметка |
