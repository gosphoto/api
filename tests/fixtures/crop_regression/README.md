# Crop regression set

Прогонять **каждый раз при изменении** `crop.py` / `compliance.py` / `baldness.py`.

| Case | Кто | Зачем |
|------|-----|--------|
| `bald_elder` | лысый дед | не раздувать верх / не ломать макушку |
| `high_hair_man` | мужчина, высокая укладка | отступ от верха причёски |
| `high_hair_girl` | девочка, объёмные волны | кейс ~3.2 мм сверху (было слишком плотно) |

Файлы в каждом кейсе:

| Файл | Назначение |
|------|------------|
| `in.jpg` | исходник (селфи / студия) |
| `white_in.jpg` | белый фон для **crop-only** (`run_crop_stage`) |
| `case.json` | ожидания по `top_margin` / `face_ratio` / pass |
| `baseline_out.jpg` | прод-out на момент фиксации набора |

## Запуск

```bash
# из backend/ (нужны deps: opencv, mediapipe, …)
pytest tests/test_crop_regression.py -v

# или скрипт с печатью метрик:
python -m tests.crop_regression_run
```

В Docker (prod image):

```bash
docker exec gosphoto-gate pytest /app/tests/test_crop_regression.py -v
```
