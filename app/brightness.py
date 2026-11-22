# app/brightness.py
from __future__ import annotations

from PIL import Image, ImageEnhance

DEFAULT_BRIGHTNESS = 1.0
MIN_BRIGHTNESS = 0.7
MAX_BRIGHTNESS = 1.35
BRIGHTNESS_EPS = 0.01


def parse_brightness(value: object) -> float:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ValueError("brightness must be a number") from e
    if not (MIN_BRIGHTNESS <= n <= MAX_BRIGHTNESS):
        raise ValueError(
            f"brightness must be between {MIN_BRIGHTNESS} and {MAX_BRIGHTNESS}"
        )
    return round(n, 4)


def is_dirty(brightness: float) -> bool:
    return abs(float(brightness) - DEFAULT_BRIGHTNESS) >= BRIGHTNESS_EPS


def apply_brightness(photo: Image.Image, brightness: float) -> Image.Image:
    factor = float(brightness)
    if abs(factor - DEFAULT_BRIGHTNESS) < 1e-6:
        return photo
    return ImageEnhance.Brightness(photo.convert("RGB")).enhance(factor)