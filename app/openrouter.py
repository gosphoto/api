"""OpenRouter Image API — white/transparent background edit, no precise crop."""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from . import config

log = logging.getLogger("gosphoto-gate")

EDIT_PROMPT = (
    "STRICT edit scope for a Russian passport / Gosuslugi photo. "
    "ALLOWED only: (1) replace background with transparent alpha or pure #FFFFFF; "
    "(2) clean the shoulder/clothing outline BELOW the neck. "
    "FORBIDDEN: any change to the head — face, eyes, brows, nose, mouth, ears, "
    "skin, pores, freckles, wrinkles, stubble, expression, makeup, hair style/color/volume. "
    "The entire head must look pixel-identical to the input photo; do not redraw, "
    "beautify, morph, smooth, relight, or re-age the face. "
    "Do not invent a new face. Keep identity 100%. "
    "Only shoulders and background may differ. Remove colour spill / halo on clothes edges."
)


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _aspect_ratio_for_model(model: str) -> str:
    """gpt-image-* accepts 1:1|3:2|2:3|auto — not 3:4 (causes HTTP 400)."""
    m = (model or "").lower()
    if "gpt-image" in m:
        return "2:3"
    return "3:4"


def build_edit_payload(image_bytes: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
    transparent = bool(config.OPENROUTER_TRANSPARENT_BG)
    model = config.OPENROUTER_IMAGE_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "prompt": EDIT_PROMPT,
        "aspect_ratio": _aspect_ratio_for_model(model),
        "output_format": "png" if transparent else "jpeg",
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        ],
    }
    if transparent:
        payload["background"] = "transparent"
    return payload


def edit_selfie(image_bytes: bytes, mime: str = "image/jpeg") -> bytes:
    if not config.OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured")

    url = f"{config.OPENROUTER_BASE_URL}/images"
    payload = build_edit_payload(image_bytes, mime)
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://gosphoto.ru",
        "X-Title": "Gosphoto",
    }

    with httpx.Client(timeout=config.OPENROUTER_TIMEOUT_SEC) as client:
        resp = client.post(url, json=payload, headers=headers)

    if resp.status_code >= 400:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text[:500]
        log.error("OpenRouter error %s: %s", resp.status_code, err_body)
        raise OpenRouterError(
            f"OpenRouter request failed ({resp.status_code})",
            status=resp.status_code,
            body=err_body,
        )

    data = resp.json()
    items = data.get("data") or []
    if not items:
        raise OpenRouterError("OpenRouter returned no image", body=data)

    item = items[0]
    b64 = item.get("b64_json")
    if not b64:
        # some providers return url
        img_url = item.get("url")
        if img_url:
            with httpx.Client(timeout=60) as client:
                r = client.get(img_url)
                r.raise_for_status()
                return r.content
        raise OpenRouterError("OpenRouter image missing b64_json/url", body=item)

    return base64.b64decode(b64)
