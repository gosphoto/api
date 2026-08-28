from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app import crop
from app.openrouter import POST_CROP_CLEANUP_PROMPT


def _jpeg(value: int, *, width: int = 60, height: int = 80) -> bytes:
    image = np.full((height, width, 3), value, np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def test_post_crop_prompt_only_cleans_background():
    prompt = POST_CROP_CLEANUP_PROMPT.lower()
    assert "#ffffff" in prompt
    assert "gray" in prompt or "grey" in prompt
    assert "background" in prompt
    assert "do not change" in prompt
    assert "face" in prompt
    assert "hair" in prompt
    assert "clothing" in prompt
    assert "framing" in prompt or "crop" in prompt
    assert "only background pixels" in prompt
    assert "do not change geometry" in prompt
    assert "do not change the face" in prompt
    assert "do not change the hairstyle" in prompt
    assert "do not change the clothing" in prompt
    assert "under no circumstances change face geometry" in prompt
    assert "ни в коем случае не меняй геометрию лица" in POST_CROP_CLEANUP_PROMPT
    assert "shorten" in prompt
    assert "widen" in prompt
    assert "stretch" in prompt
    assert "warp" in prompt
    assert "facial proportions" in prompt
    assert "look very carefully" in prompt
    assert "braid" in prompt or "pigtail" in prompt
    assert "смотри внимательнее" in POST_CROP_CLEANUP_PROMPT
    assert "косичками" in POST_CROP_CLEANUP_PROMPT
    assert "рядом с лицом" in POST_CROP_CLEANUP_PROMPT
    assert "hard lock" in prompt
    assert "do not zoom" in prompt
    assert "top margin" in prompt
    assert "не сдвигай голову" in POST_CROP_CLEANUP_PROMPT
    assert "не меняй размер лица" in POST_CROP_CLEANUP_PROMPT
    assert "не трогай кроп" in POST_CROP_CLEANUP_PROMPT


def test_cleanup_runs_before_final_soft_whitening(monkeypatch):
    source = np.full((100, 70, 3), 241, np.uint8)
    calls: list[str] = []

    monkeypatch.setattr(crop.config, "POST_CROP_CLEANUP_ENABLED", True)
    monkeypatch.setattr(crop.config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        crop.config, "POST_CROP_CLEANUP_MODEL", "google/gemini-2.5-flash-image"
    )

    def fake_edit(image_bytes, mime, *, model, prompt):
        calls.append("model")
        assert model == "google/gemini-2.5-flash-image"
        assert prompt == POST_CROP_CLEANUP_PROMPT
        return _jpeg(255)

    def fake_whiten(image, tol=52, *, soften=True):
        calls.append("soften" if soften else "no-soften")
        return image

    monkeypatch.setattr(crop, "edit_selfie_riverflow", fake_edit)
    monkeypatch.setattr(crop, "force_white_background", fake_whiten)

    out, meta = crop._finalize_crop(source)

    assert calls == ["model", "soften"]
    assert out.shape == source.shape
    assert meta == {
        "applied": True,
        "model": "google/gemini-2.5-flash-image",
    }


def test_cleanup_failure_falls_back_to_selected_crop(monkeypatch):
    source = np.full((100, 70, 3), 241, np.uint8)

    monkeypatch.setattr(crop.config, "POST_CROP_CLEANUP_ENABLED", True)
    monkeypatch.setattr(crop.config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(crop.config, "POST_CROP_CLEANUP_MODEL", "same-model")
    monkeypatch.setattr(
        crop,
        "edit_selfie_riverflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("model down")),
    )
    monkeypatch.setattr(
        crop,
        "force_white_background",
        lambda image, tol=52, *, soften=True: image,
    )

    out, meta = crop._finalize_crop(source)

    np.testing.assert_array_equal(out, source)
    assert meta["applied"] is False
    assert meta["model"] == "same-model"
    assert "model down" in meta["error"]


def test_crop_stage_finalizes_only_selected_candidate(monkeypatch):
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "crop_regression"
        / "high_hair_man"
        / "white_in.jpg"
    )
    source = cv2.imread(str(fixture))
    assert source is not None
    calls = 0

    def fake_finalize(image):
        nonlocal calls
        calls += 1
        return image, {"applied": False, "reason": "test"}

    monkeypatch.setattr(crop, "_finalize_crop", fake_finalize)

    _, metrics, _ = crop.run_crop_stage(source)

    assert calls == 1
    assert metrics["post_crop_cleanup"]["reason"] == "test"


def test_crop_stage_rejects_cleanup_that_breaks_compliance(monkeypatch):
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "crop_regression"
        / "high_hair_man"
        / "white_in.jpg"
    )
    source = cv2.imread(str(fixture))
    assert source is not None

    def fake_finalize(image):
        return np.full_like(image, 255), {
            "applied": True,
            "model": "unsafe-cleaner",
        }

    monkeypatch.setattr(crop, "_finalize_crop", fake_finalize)

    out, metrics, compliance = crop.run_crop_stage(source)

    assert compliance["pass"] is True
    assert not np.all(out == 255)
    cleanup = metrics["post_crop_cleanup"]
    assert cleanup["applied"] is False
    assert cleanup["rejected"] is True
    assert cleanup["reason"] == "compliance_regression"
