"""Crop prefers Pose shoulder roll when available; else eye roll."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from app.crop import _crop_once
from app.torso import ShoulderRoll

FIX = Path(__file__).resolve().parent / "fixtures" / "crop_regression" / "high_hair_man"


@pytest.fixture
def white_portrait():
    src = FIX / "white_in.jpg"
    assert src.is_file(), f"missing {src}"
    bgr = cv2.imread(str(src))
    assert bgr is not None
    return bgr


def test_crop_uses_shoulder_roll_when_available(monkeypatch, white_portrait):
    monkeypatch.setattr(
        "app.crop.measure_shoulder_roll",
        lambda _bgr: ShoulderRoll(deg=7.5, metrics={"reason": "ok"}),
    )
    _out, metrics = _crop_once(
        white_portrait, crown_factor=0.45, face_ratio=0.79, top_margin=0.10
    )
    assert metrics["roll_source"] == "shoulders"
    assert metrics["shoulder_roll_deg"] == 7.5
    assert metrics["roll_corrected_deg"] == 7.5
    assert "eye_roll_deg" in metrics


def test_crop_falls_back_to_eyes_when_no_shoulders(monkeypatch, white_portrait):
    monkeypatch.setattr("app.crop.measure_shoulder_roll", lambda _bgr: None)
    _out, metrics = _crop_once(
        white_portrait, crown_factor=0.45, face_ratio=0.79, top_margin=0.10
    )
    assert metrics["roll_source"] == "eyes"
    assert metrics["shoulder_roll_deg"] is None
    assert metrics["roll_corrected_deg"] == metrics["eye_roll_deg"]
