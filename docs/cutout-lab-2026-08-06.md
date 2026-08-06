# Cutout lab — белый фон без искажения лица

> Проверено: 2026-08-06 · источники: локальные эксперименты (ONNX + MediaPipe) на `tmp-smoke/site-smoke/00-input.jpg`

## Вердикт

**Лучший рабочий рецепт:** `silueta` ONNX + morph-close (k=41) + face-lock + `restore_face_from_original`.  
Лицо не перерисовывается; силуэт фигуры можно «заливать» close, чтобы убрать дыры у шеи/плеч.

**С чем идём в прод:** `EDIT_CUTOUT=silueta` (fallback `u2netp` → MediaPipe).

## Как смотреть артефакты

| Файл | Что внутри |
|------|------------|
| [`tmp-smoke/cutout-lab/board-top8.jpg`](../tmp-smoke/cutout-lab/board-top8.jpg) | Round1: top‑8 по метрикам |
| [`tmp-smoke/cutout-lab/r2/board-best.jpg`](../tmp-smoke/cutout-lab/r2/board-best.jpg) | Round2: face-protect (метрики ≠ визуал) |
| [`tmp-smoke/cutout-lab/r4/board-best.jpg`](../tmp-smoke/cutout-lab/r4/board-best.jpg) | Round4: INPUT / PROD / BEST |
| [`tmp-smoke/cutout-lab/r4/champion_crop.jpg`](../tmp-smoke/cutout-lab/r4/champion_crop.jpg) | Чемпион 35×45 |
| [`tmp-smoke/cutout-lab/final/board-final.jpg`](../tmp-smoke/cutout-lab/final/board-final.jpg) | Итоговое сравнение |
| [`tmp-smoke/cutout-lab/scores.json`](../tmp-smoke/cutout-lab/scores.json) | Сырые скоры Round1 |
| [`scripts/cutout_lab.py`](../scripts/cutout_lab.py) | Матрица вариантов |

## Раунды

### Round 1 — модели × post × face_restore

Модели: `u2netp`, `u2net`, `silueta`, `isnet-general-use`.  
Post: `soft1/soft2/dist/guided`. С/без face restore.

| Наблюдение | Вывод |
|------------|--------|
| `isnet_*` часто **режет лицо** белыми пятнами | не брать для прод |
| Метрика `shoulder_canny` низкая ≠ хорошо (лицо «съедено») | нужен визуальный QA |
| Лучшие по face+fringe среди целых: `u2net_soft2_fr`, `silueta_soft2_*` | база для R2/R4 |

### Round 2 — face-protect в маске

Форс `max(mask, face_oval)` + body erode + purge border blobs.

| Итог | Проблема |
|------|----------|
| Метрики fringe ↓ | Визуально оставались **дыры фона у шеи/плеч** (кухня) |
| «BEST» по числам `silueta_soft2_e3_pg` | Хуже PROD глазами |

### Round 3 — hard threshold

Жёсткий thr + feather. Fringe_p90 вырос; leftover-метрика шумела (считала кожу).

### Round 4 — morph-close (победитель)

`silueta`, thr=0.52, elliptical close k=41, erode=2, face-lock, face restore вокруг whitening.

| vs PROD (`u2netp` soft) | Эффект |
|-------------------------|--------|
| Края | заметно **глаже**, меньше лесенки |
| Дыры у шеи | **закрываются** close |
| Лицо | без generative-перерисовки (restore) |

Чемпион: `r4_silueta_t52_c41_close` → `tmp-smoke/cutout-lab/r4/champion_crop.jpg`.

### Финальная попытка purge бликов

Доп. kill ярких пикселей на контуре → **белые пятна на лбу/шее** (блики = false positive). **Откатили.**

## Не брать

- `isnet-general-use` на этом селфи — порча лица  
- Чистый convex hull на весь кадр — раздувает силуэт  
- Агрессивный edge-purge по luma — ест блики кожи  
- OpenRouter generative edit в `/api/process` — искажает лицо  

## Прод-настройки

```text
EDIT_CUTOUT=silueta
SILUETA_MODEL_PATH=/app/models/silueta.onnx
U2NETP_MODEL_PATH=/app/models/u2netp.onnx   # fallback
```

VPS ~1.8 GiB: `silueta` ~42 MB — ок; полный `u2net` ~168 MB лучше не держать в контейнере без нужды.

## История правок

| Дата | Что изменили |
|------|----------------|
| 2026-08-06 | Лаб R1–R4, чемпион silueta+close, док + прод default |
