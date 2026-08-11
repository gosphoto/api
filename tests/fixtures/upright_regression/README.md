# Upright regression set

Прогонять **каждый раз при изменении** `gate.py` (`upright_image` / `_pose_score`).

| Case | Кто | Зачем |
|------|-----|--------|
| `inverted_selfie` | оплаченный кейс 2026-08-11 (`IMG_2137`) | MediaPipe врёт `upright=true` на перевёрнутом лице; без `face_center_y` gate выбирал 0° и отдавал invert в edit |

Файлы в каждом кейсе:

| Файл | Назначение |
|------|------------|
| `in.jpg` | вход (здесь — уже перевёрнутый кадр, как ушёл в process на проде) |
| `case.json` | ожидания `rotation_deg` / `face_center_y` |

## Запуск

```bash
export GATE_MODEL_PATH=$PWD/models/face_landmarker.task PYTHONPATH=$PWD
pytest tests/test_upright_regression.py tests/test_gate_upright.py -v
```
