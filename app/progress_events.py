"""Synthetic process-progress events for SSE while Riverflow runs."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

PROGRESS_INTERVAL_SEC = 4.0

# Product copy: enough unique lines for ~1–3 min at 4s without hanging on one label.
PROGRESS_STEPS: list[dict[str, str]] = [
    {"text": "Проверяем лицо и качество снимка…"},
    {"text": "Смотрим резкость и освещение…"},
    {"text": "Убеждаемся, что в кадре один человек…"},
    {"text": "Готовим кадр к обработке фона…"},
    {"text": "Убираем фон, делаем чисто белый…"},
    {"text": "Аккуратно обходим контур волос…"},
    {"text": "Убираем ореол и цветной контур у плеч…"},
    {"text": "Проверяем, что фон без теней…"},
    {"text": "Сохраняем черты лица один в один…"},
    {"text": "Кадрируем под формат 35×45 мм…"},
    {"text": "Выравниваем положение головы…"},
    {"text": "Проверяем отступы по ГОСТ / п. 34.3…"},
    {"text": "Готовим JPEG для загрузки на Госуслуги…"},
    {"text": "Собираем лист 10×15 с четырьмя фото…"},
    {"text": "Финальная проверка файла…"},
    {"text": "Ещё немного — нейросеть обрабатывает фон…"},
    {"text": "Почти готово, дожимаем качество…"},
    {"text": "Сохраняем результат…"},
]

_TAIL = [
    "Нейросеть ещё работает над фоном…",
    "Это занимает до пары минут — не закрывайте вкладку…",
    "Доводим края и белый фон…",
    "Совсем скоро будет готово…",
]


def next_progress(index: int) -> dict[str, Any]:
    """Return progress payload for tick ``index`` (0-based).

    Percent asymptotes toward 99 and never hits 100 (reserved for client ``done``).
    After the main list, cycles tail phrases so the UI never freezes on one line.
    """
    n = len(PROGRESS_STEPS)
    if index < n:
        text = PROGRESS_STEPS[index]["text"]
        # Spread main list across ~5% … ~90%
        pct = 5.0 + (85.0 * index / max(n - 1, 1))
    else:
        text = _TAIL[(index - n) % len(_TAIL)]
        # Slow crawl 90 → 99
        over = index - n + 1
        pct = 90.0 + 9.0 * (1.0 - math.exp(-over / 12.0))

    pct_i = int(min(99, max(1, round(pct))))
    if pct_i >= 100:
        pct_i = 99
    return {
        "event": "progress",
        "index": index,
        "text": text,
        "pct": pct_i,
    }


def format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


async def iter_process_sse(
    *,
    work: Callable[[], Awaitable[dict[str, Any]]],
    interval_sec: float = PROGRESS_INTERVAL_SEC,
) -> AsyncIterator[str]:
    """Yield SSE frames every ``interval_sec`` until ``work`` finishes.

    Final frame is ``done`` when ``ok`` is true, otherwise ``error``.
    """
    task = asyncio.create_task(work())
    index = 0
    try:
        while not task.done():
            ev = next_progress(index)
            yield format_sse("progress", ev)
            index += 1
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=interval_sec)
            except asyncio.TimeoutError:
                continue
            except Exception:
                # Work failed; fall through to task.result() error handling.
                break

        try:
            result = task.result()
        except Exception as exc:  # noqa: BLE001 — surfaced to client as error event
            yield format_sse(
                "error",
                {
                    "ok": False,
                    "message": str(exc) or exc.__class__.__name__,
                },
            )
            return

        if not result.get("ok", False):
            yield format_sse(
                "error",
                {
                    "ok": False,
                    "stage": result.get("stage"),
                    "reason": result.get("reason"),
                    "message": result.get("message") or "Не удалось обработать фото",
                    "metrics": result.get("metrics"),
                },
            )
            return

        yield format_sse("done", result)
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
