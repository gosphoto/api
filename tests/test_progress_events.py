"""Unit tests for synthetic process-progress events (SSE ticker)."""

from __future__ import annotations

import asyncio
import json
import time

from app import progress_events as pe


def test_progress_steps_has_at_least_12_unique_texts():
    texts = [s["text"] for s in pe.PROGRESS_STEPS]
    assert len(texts) >= 12
    assert len(set(texts)) >= 12


def test_progress_steps_are_non_empty_russian():
    for step in pe.PROGRESS_STEPS:
        assert isinstance(step["text"], str) and step["text"].strip()
        # Cyrillic present in product copy
        assert any("а" <= c.lower() <= "я" or c in "ё" for c in step["text"].lower())


def test_next_progress_advances_percent_monotonically_early():
    pcts = [pe.next_progress(i)["pct"] for i in range(len(pe.PROGRESS_STEPS))]
    assert pcts[0] >= 1
    assert pcts[-1] <= 95
    assert all(a <= b for a, b in zip(pcts, pcts[1:]))


def test_next_progress_never_reaches_100_before_done():
    for i in range(200):
        ev = pe.next_progress(i)
        assert ev["pct"] < 100
        assert ev["pct"] <= 99


def test_next_progress_cycles_text_after_list_exhausted():
    n = len(pe.PROGRESS_STEPS)
    a = pe.next_progress(n)["text"]
    b = pe.next_progress(n + 1)["text"]
    # Still returns valid cycling copy; not empty / not frozen forever on one index
    assert a and b
    # Over a window of cycle length we see more than one distinct text
    window = {pe.next_progress(n + i)["text"] for i in range(n)}
    assert len(window) >= 3


def test_next_progress_includes_index_and_event_type():
    ev = pe.next_progress(0)
    assert ev["event"] == "progress"
    assert ev["index"] == 0
    assert "text" in ev and "pct" in ev


def test_format_sse_progress_frame():
    frame = pe.format_sse("progress", {"text": "Проверяем…", "pct": 12, "index": 0})
    assert frame.startswith("event: progress\n")
    assert "data: " in frame
    assert frame.endswith("\n\n")
    data_line = [ln for ln in frame.split("\n") if ln.startswith("data: ")][0]
    payload = json.loads(data_line[len("data: ") :])
    assert payload["pct"] == 12
    assert payload["text"] == "Проверяем…"


def test_format_sse_done_frame():
    frame = pe.format_sse("done", {"ok": True, "result_id": "abc"})
    assert "event: done\n" in frame
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["result_id"] == "abc"


def test_progress_ticker_emits_every_interval_until_work_done():
    """Async generator yields progress on interval, then done — no hang at last %."""

    async def _run():
        async def quick_work():
            await asyncio.sleep(0.25)
            return {"ok": True, "result_id": "rid-1"}

        frames: list[str] = []
        t0 = time.monotonic()
        async for frame in pe.iter_process_sse(
            work=quick_work,
            interval_sec=0.08,
        ):
            frames.append(frame)
        elapsed = time.monotonic() - t0

        events = []
        for frame in frames:
            assert frame.endswith("\n\n")
            name = frame.split("\n", 1)[0].removeprefix("event: ").strip()
            data = json.loads(frame.split("data: ", 1)[1].strip())
            events.append((name, data))

        progress_events = [e for e in events if e[0] == "progress"]
        done_events = [e for e in events if e[0] == "done"]

        assert len(progress_events) >= 2  # several ticks before ~0.25s work
        assert len(done_events) == 1
        assert done_events[0][1]["result_id"] == "rid-1"
        texts = [e[1]["text"] for e in progress_events]
        assert len(set(texts)) >= 2
        assert elapsed < 2.0

    asyncio.run(_run())


def test_progress_ticker_emits_error_when_work_raises():
    async def _run():
        async def boom():
            await asyncio.sleep(0.05)
            raise RuntimeError("riverflow_failed")

        frames = [
            frame
            async for frame in pe.iter_process_sse(work=boom, interval_sec=0.05)
        ]
        names = [f.split("\n", 1)[0] for f in frames]
        assert any(n == "event: error" for n in names)
        err = next(f for f in frames if f.startswith("event: error"))
        payload = json.loads(err.split("data: ", 1)[1].strip())
        assert payload["ok"] is False
        assert "riverflow_failed" in payload["message"]

    asyncio.run(_run())


def test_progress_ticker_gate_reject_as_error_event():
    async def _run():
        async def gate_fail():
            return {
                "ok": False,
                "stage": "gate",
                "reason": "blur",
                "message": "Слишком размыто",
            }

        frames = [
            frame
            async for frame in pe.iter_process_sse(work=gate_fail, interval_sec=0.05)
        ]
        err = next(f for f in frames if f.startswith("event: error"))
        payload = json.loads(err.split("data: ", 1)[1].strip())
        assert payload["stage"] == "gate"
        assert payload["ok"] is False

    asyncio.run(_run())
