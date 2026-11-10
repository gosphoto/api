"""Unit tests for Replicate background-removal client."""

from __future__ import annotations

import json

import httpx
import numpy as np
import cv2
import pytest

from app import config
from app import replicate_bg as rb


def _png_rgba(w: int = 8, h: int = 8) -> bytes:
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, :3] = (40, 80, 120)
    img[:, :, 3] = 255
    img[:2, :, 3] = 0
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_remove_background_happy_path(monkeypatch):
    monkeypatch.setattr(config, "REPLICATE_API_TOKEN", "r8_test")
    monkeypatch.setattr(config, "REPLICATE_BG_VERSION", "ver-abc")
    monkeypatch.setattr(config, "REPLICATE_TIMEOUT_SEC", 5.0)
    monkeypatch.setattr(config, "REPLICATE_POLL_INTERVAL_SEC", 0.01)

    png = _png_rgba()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if request.url.path == "/v1/files":
            return httpx.Response(
                201,
                json={
                    "id": "file1",
                    "urls": {"get": "https://api.replicate.com/v1/files/file1"},
                },
            )
        if request.url.path == "/v1/predictions" and request.method == "POST":
            body = json.loads(request.content.decode())
            assert body["version"] == "ver-abc"
            assert body["input"]["background_type"] == "rgba"
            assert body["input"]["threshold"] == 0
            return httpx.Response(
                201,
                json={"id": "pred1", "status": "starting", "output": None},
            )
        if request.url.path == "/v1/predictions/pred1":
            return httpx.Response(
                200,
                json={
                    "id": "pred1",
                    "status": "succeeded",
                    "output": "https://replicate.delivery/out.png",
                    "metrics": {"predict_time": 1.5, "total_time": 1.7},
                },
            )
        if request.url.path == "/out.png" or str(request.url).endswith("/out.png"):
            return httpx.Response(200, content=png)
        return httpx.Response(404, json={"detail": f"unexpected {request.url}"})

    transport = httpx.MockTransport(handler)

    class FakeClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.setdefault("base_url", "https://api.replicate.com")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rb.httpx, "Client", FakeClient)
    # download hits absolute replicate.delivery URL — FakeClient has no base for that.
    # Patch download_bytes path by making MockTransport match host-agnostic paths above;
    # absolute URL /out.png: MockTransport still calls handler with full URL.

    out, meta = rb.remove_background(b"fake-jpeg-bytes", mime="image/jpeg")
    assert out == png
    assert meta["ok"] is True
    assert meta["applied"] is True
    assert meta["prediction_id"] == "pred1"
    assert meta["predict_time"] == 1.5
    assert any(c.startswith("POST /v1/files") for c in calls)
    assert any(c.startswith("POST /v1/predictions") for c in calls)


def test_remove_background_requires_token(monkeypatch):
    monkeypatch.setattr(config, "REPLICATE_API_TOKEN", "")
    with pytest.raises(rb.ReplicateBgError, match="REPLICATE_API_TOKEN"):
        rb.remove_background(b"x")


def test_remove_background_prediction_failed(monkeypatch):
    monkeypatch.setattr(config, "REPLICATE_API_TOKEN", "r8_test")
    monkeypatch.setattr(config, "REPLICATE_BG_VERSION", "ver-abc")
    monkeypatch.setattr(config, "REPLICATE_TIMEOUT_SEC", 5.0)
    monkeypatch.setattr(config, "REPLICATE_POLL_INTERVAL_SEC", 0.01)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/files":
            return httpx.Response(
                201,
                json={"urls": {"get": "https://api.replicate.com/v1/files/f"}},
            )
        if request.url.path == "/v1/predictions" and request.method == "POST":
            return httpx.Response(201, json={"id": "pred-fail", "status": "starting"})
        if request.url.path == "/v1/predictions/pred-fail":
            return httpx.Response(
                200,
                json={"id": "pred-fail", "status": "failed", "error": "boom", "output": None},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    class FakeClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            kwargs.setdefault("base_url", "https://api.replicate.com")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(rb.httpx, "Client", FakeClient)
    with pytest.raises(rb.ReplicateBgError, match="failed"):
        rb.remove_background(b"jpeg")
