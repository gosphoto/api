"""Hard-reject black-and-white uploads; color portraits must still pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.gate import validate_image

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "grayscale_gate"

BW = [
    "bw_stepa.jpg",
    "bw_olga_pasha.jpg",
    "bw_2.jpg",
    "bw_3.jpg",
]
COLOR = [
    "color_1.jpg",
    "color_2.jpg",
    "color_3.jpg",
    "color_4.jpg",
    "color_5.jpg",
]

GRAYSCALE_MESSAGE = "Черно белое фото не допускается"


def _load(name: str) -> bytes:
    path = FIXTURES / name
    assert path.is_file(), f"missing fixture {path}"
    return path.read_bytes()


@pytest.mark.parametrize("name", BW)
def test_grayscale_photos_are_rejected(name: str):
    gate = validate_image(_load(name))
    assert gate.ok is False
    assert gate.reason == "grayscale"
    assert gate.message == GRAYSCALE_MESSAGE


@pytest.mark.parametrize("name", COLOR)
def test_color_photos_are_accepted(name: str):
    gate = validate_image(_load(name))
    assert gate.reason != "grayscale"
    assert gate.ok is True
    assert gate.reason is None
