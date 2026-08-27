"""Edit is always Gemini/Riverflow — never skip because the selfie looks studio-ready."""

from unittest.mock import MagicMock, patch

import numpy as np

from app import config
from app import main as main_mod
from app.readiness import ReadinessResult


def test_skip_edit_if_ready_defaults_off():
    assert config.SKIP_EDIT_IF_READY is False


def test_passport_stages_always_calls_edit_even_if_studio_ready(monkeypatch):
    monkeypatch.setattr(main_mod.config, "SKIP_EDIT_IF_READY", True)
    ready = ReadinessResult(ready=True, reason="studio_white", scores={})
    edited = np.full((40, 32, 3), 255, np.uint8)
    cropped = np.full((50, 40, 3), 255, np.uint8)
    edit_calls = []

    def fake_edit(data, mime="image/jpeg"):
        edit_calls.append((data, mime))
        return edited, {"cutout": "openrouter_edit", "model": "openai/gpt-5-image-mini"}

    monkeypatch.setattr(main_mod, "_decode_image", lambda _d: edited)
    monkeypatch.setattr(main_mod, "assess_readiness", lambda _im: ready)
    monkeypatch.setattr(main_mod, "run_edit_stage", fake_edit)
    monkeypatch.setattr(
        main_mod,
        "run_crop_stage",
        lambda *_a, **_k: (cropped, {"width": 40, "height": 50}, {"pass": True, "checks": {}}),
    )
    monkeypatch.setattr(main_mod, "encode_jpeg", lambda *_a, **_k: b"JPEG")
    monkeypatch.setattr(
        main_mod,
        "_print_payload",
        lambda *_a, **_k: (b"PRINT", {"width": 1, "height": 1, "dpi": 300, "copies": 4, "bytes": 1, "size_cm": [10, 15], "mime": "image/jpeg", "layout": "2x2"}),
    )

    out = main_mod._run_passport_stages(b"selfie", mime="image/jpeg")
    assert out["ok"] is True
    assert out["skipped_edit"] is False
    assert edit_calls == [(b"selfie", "image/jpeg")]
    assert out["edit_meta"]["cutout"] == "openrouter_edit"
