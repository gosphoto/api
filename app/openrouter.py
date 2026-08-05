"""OpenRouter Image API — cheap edit (white background), no precise crop."""

from __future__ import annotations

import base64
import logging
from typing import Any

import httpx

from . import config

log = logging.getLogger("gosphoto-gate")

EDIT_PROMPT = (
    "Create a Russian biometric passport ID photo from this selfie. "
    "Plain pure white seamless background, even soft lighting, no shadows on the wall. "
    "Keep the person's exact face identity, skin, hair, and facial features unchanged — "
    "no beauty retouch, no face morphing, no makeup. "
    "Frontal head-and-shoulders portrait, neutral expression, mouth closed, eyes open. "
    "Document photo style only."
)


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


def _data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def edit_selfie(image_bytes: bytes, mime: str = "image/jpeg") -> bytes:
    if not config.OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured")

    url = f"{config.OPENROUTER_BASE_URL}/images"
    payload: dict[str, Any] = {
        "model": config.OPENROUTER_IMAGE_MODEL,
        "prompt": EDIT_PROMPT,
        "aspect_ratio": "3:4",
        "output_format": "jpeg",
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        ],
    }
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
