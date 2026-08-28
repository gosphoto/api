"""OpenRouter Image API — Riverflow white-bg edit (+ legacy /images helper)."""

from __future__ import annotations

import base64
import logging
import re
from typing import Any

import httpx

from . import config

log = logging.getLogger("gosphoto-gate")

EDIT_PROMPT = (
    "STRICT edit scope for a Russian passport / Gosuslugi photo. "
    "ALLOWED only: (1) replace background with transparent alpha or pure #FFFFFF; "
    "(2) clean the shoulder/clothing outline BELOW the neck; "
    "(3) if the source is dark, underexposed, noisy or soft — mild exposure lift "
    "and light denoise on the face, not a beauty filter. "
    "FORBIDDEN: any change to the head — face identity, eyes, brows, nose, mouth, ears, "
    "skin color, pores, freckles, wrinkles, stubble, expression, makeup, hair style/color/volume. "
    "Do not redraw, beautify, morph, beauty-smooth, or re-age the face. "
    "Do not invent a new face. Keep identity 100%. "
    "If the source is already well-lit and sharp, do not relight or retouch the face. "
    "Jewelry/accessories only if present on the input — do not add earrings, "
    "nose rings, piercings, chains, or pendants. If the selfie has no earrings, "
    "keep bare ears with no earrings. "
    "Only shoulders, background, and this optional exposure rescue may differ. "
    "Remove colour spill / halo on clothes edges."
)

# Resume / LinkedIn-style portrait (separate SKU).
RESUME_SUIT_PROMPT = (
    "Professional resume / LinkedIn headshot from this selfie. "
    "If the person wears religious clothing or a religious head covering "
    "(hijab, khimar, tichel, turban, kippah, nun's veil, abaya, modest religious "
    "dress) with the full face oval clearly visible: KEEP that clothing and "
    "covering exactly — do NOT replace with a business suit, blazer, shirt, or tie. "
    "Otherwise dress the person in a stylish modern business suit "
    "(well-fitted blazer, dress shirt, subtle tie optional if it fits the look). "
    "Apply light natural retouch: even skin tone, soft shine reduction, "
    "keep pores and real texture — no plastic skin, no heavy beauty filter, "
    "no age change. "
    "CRITICAL: preserve exact face identity, age, hair style/color, expression, "
    "eye color, facial proportions from the input. "
    "Framing: upper body / shoulders visible, face centered, soft studio lighting, "
    "neutral light-gray or soft off-white seamless backdrop. "
    "No watermarks, text, logos, or frames. Photorealistic, high quality."
)

# Main process edit = resume look, but pure white document background + clean face spots.
GOSUSLUGI_EDIT_PROMPT = (
    "Professional resume / LinkedIn headshot from this selfie. "
    "CLOTHING: if the person wears religious clothing or a religious head covering "
    "(hijab, khimar, tichel, turban, kippah, nun's veil, abaya, modest religious "
    "dress) and the full face oval is clearly visible: KEEP that clothing type "
    "and covering — do NOT replace with a business suit, blazer, shirt, or tie. "
    "Face oval must stay fully open (no niqab/burqa covering the face). "
    "If the hijab / religious covering / religious clothing is light / white / "
    "off-white / pale-grey and would vanish on a white backdrop: recolor it to "
    "a dark solid (navy, charcoal, or dark blue) — keep the same hijab/abaya "
    "shape and drape, only the color changes. Never leave a white hijab on "
    "#FFFFFF. "
    "If there is no religious clothing or covering, dress an adult in a stylish "
    "modern business suit "
    "(well-fitted blazer, dress shirt, subtle tie optional if it fits the look). "
    "GRAY / grey shirt collar on #FFFFFF: never let a gray or grey shirt collar "
    "touch or blend into the white backdrop. Either (1) keep the blazer / jacket "
    "lapels so they fully separate the shirt collar from the background, or "
    "(2) recolor that gray/grey dress shirt to a light-blue business shirt "
    "(голубая деловая) so the collar reads clearly against #FFFFFF. "
    "If the person is a child, or wears a light / white / off-white / pale-grey "
    "t-shirt, tank top, or undershirt that would vanish on a white backdrop: "
    "do NOT keep pale clothes. Children: replace with a dark solid child's "
    "t-shirt (navy, charcoal, or dark blue) — never a business suit, never pale "
    "clothes on #FFFFFF. Adults in a light tank/tee (and not in religious "
    "clothing): still the business suit. "
    "Face exposure: keep the input face brightness and skin tone unchanged — "
    "do not brighten, bleach, overexpose, flash-relight, or wash out the skin; "
    "no high-key studio face lighting; cheeks/forehead must not turn chalky white. "
    "Retouch only local temporary blemishes and redness patches without lifting "
    "overall exposure — keep pores and real texture — no plastic skin, no heavy "
    "beauty filter, no age change. "
    "FORBIDDEN skin marks: do not invent, add, densify, or scatter moles, freckles, "
    "birthmarks, age spots, or any new skin dots/patches that are not clearly "
    "visible on the source face; if the input has few or no freckles/moles, the "
    "output must match — zero invented spots. "
    "CRITICAL: preserve exact face identity, age, hair style/color, "
    "eye color, facial proportions from the input. "
    "Gosuslugi / RF passport rules: face must match the person's age; face fully "
    "open (hair must not cover eyes, brows, or face oval); gaze straight into the "
    "camera lens; neutral closed-mouth expression — no smile, no laugh, no grimace "
    "(if the input smiles, relax the mouth to neutral — do not change identity). "
    "Posture: seat the person upright and straight — perfectly level shoulders "
    "(left and right at the same height), head upright with no tilt/roll, "
    "spine vertical, face looking straight into the camera (yaw≈0). "
    "Correct any slanted selfie posture. "
    "If the person wears eyeglasses in the input: keep the same glasses; remove "
    "all lens glare, reflections, hotspots and flash bounce; eyes must be fully "
    "visible through clear lenses; frames must not cover or hide the eyes; "
    "no sunglasses, no tinted lenses; do not invent glasses if the input has none. "
    "No secular headwear, hats, caps, headbands, decorative scarves, or "
    "service/military uniforms. Religious head covering and religious clothing "
    "must be kept when the full face oval is clearly visible "
    "(e.g. hijab with open face) — never strip them for a suit. "
    "Subject alone in frame: no other people, no toys, no pacifiers, no foreign "
    "objects in hands or near the face (same rules for children as for adults). "
    "Framing: upper body / shoulders visible, face centered; "
    "keep natural face lighting from the input (not a bright studio key light). "
    "Background: pure seamless #FFFFFF white only — no gray, no off-white, "
    "no shadows, vignette, or room props. "
    "MUST erase every isolated flyaway / stray hair that sticks out of the hair "
    "mass (temples, crown, sides, neck). Those wisps leave leftover original-wall "
    "/ gray halo on #FFFFFF — remove them completely or fade them until invisible. "
    "Zero stray strands, zero halo, zero leftover wall pixels around hair. "
    "Do not restyle the hair mass, do not tuck it behind the ears, do not gather "
    "it, do not cut the main hair, do not change length, color, or volume. "
    "No watermarks, text, logos, or frames. Photorealistic, high quality."
)

POST_CROP_CLEANUP_PROMPT = (
    "EDIT ONLY BACKGROUND PIXELS. "
    "UNDER NO CIRCUMSTANCES change face geometry — "
    "ни в коем случае не меняй геометрию лица: "
    "do not shorten, widen, round, slim, stretch, warp, or reshape the face, "
    "jaw, cheeks, chin, eyes, nose, mouth, or facial proportions. "
    "Do not change geometry. Do not change the face. Do not change the hairstyle. "
    "Do not change the clothing. "
    "Check whether any gray or grey background remains behind the person. "
    "Especially control the background behind hair or braids — "
    "особенно контролируй фон за волосами или косичками. "
    "The entire background must be one uniform pure white #FFFFFF only. "
    "Remove only gray areas, shadows, seams, halos, and remnants of the previous "
    "background. Do not change the person: preserve the exact face, identity, "
    "skin, hair, ears, neck, clothing, shoulders, colors, proportions, sharpness, "
    "position, framing, and 35x45 crop. Do not retouch, redraw, relight, beautify, "
    "recolor, or add anything. Output the same photo with only the background "
    "cleaned to solid #FFFFFF."
)

GOSUSLUGI_SCORING_PROMPT = (
    "Prefer a pure #FFFFFF seamless backdrop (not gray); keep face identity and "
    "natural skin exposure (no bleach/overexpose); no invented moles/freckles; "
    "open face, MUST erase isolated flyaway wisps and leftover wall halo "
    "(do not restyle or tuck hair); "
    "neutral no-smile expression, gaze at camera; "
    "adults: stylish modern business suit UNLESS religious clothing/covering "
    "(keep hijab/abaya — no suit; if white/pale hijab recolor navy/charcoal); "
    "gray/grey shirt collar must not touch #FFFFFF — separate with blazer "
    "lapels or recolor shirt to light-blue business; "
    "children: dark child's t-shirt "
    "(navy/charcoal) — no pale/white tank or tee on white bg; "
    "upright straight posture with perfectly level "
    "shoulders and no head tilt; if glasses present — zero lens glare, eyes fully "
    "visible under frames; no hats/uniforms (keep religious cover + clothing "
    "if face oval open); "
    "alone in frame, no toys/objects; light local spot cleanup only — "
    "without plastic skin or heavy beauty filter."
)

GOSUSLUGI_SCORING_RUBRIC: list[dict[str, Any]] = [
    {
        "key": "white_bg",
        "label": "Pure white background",
        "description": "Backdrop is solid #FFFFFF with no shadows, vignette, or room objects.",
        "weight": 0.3,
    },
    {
        "key": "identity",
        "label": "Identity fidelity",
        "description": "Same face, age, skin tone/exposure and texture, hair as the input; no bleach, no invented freckles/moles, no beautify; business suit OK unless religious clothing (keep it).",
        "weight": 0.3,
    },
    {
        "key": "hair_integrity",
        "label": "Hair without white holes",
        "description": "No swiss-cheese / salt-pepper white gaps inside the hair mass.",
        "weight": 0.25,
    },
    {
        "key": "ear_edges",
        "label": "Clean ear edges",
        "description": "No wall crumbs behind ears; cartilage preserved.",
        "weight": 0.15,
    },
]

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


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://gosphoto.ru",
        "X-Title": "Gosphoto",
    }


def build_edit_payload(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    transparent = bool(config.OPENROUTER_TRANSPARENT_BG)
    model = model or config.OPENROUTER_IMAGE_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt or EDIT_PROMPT,
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


def _decode_b64_image(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def _extract_image_bytes_from_images_api(data: dict[str, Any]) -> bytes:
    items = data.get("data") or []
    if not items:
        raise OpenRouterError("OpenRouter returned no image", body=data)
    item = items[0]
    b64 = item.get("b64_json")
    if b64:
        return _decode_b64_image(b64)
    img_url = item.get("url")
    if img_url:
        with httpx.Client(timeout=60) as client:
            r = client.get(img_url)
            r.raise_for_status()
            return r.content
    raise OpenRouterError("OpenRouter image missing b64_json/url", body=item)


def _extract_image_bytes_from_chat(data: dict[str, Any]) -> bytes:
    """Parse chat/completions multimodal image response."""
    choices = data.get("choices") or []
    if not choices:
        raise OpenRouterError("OpenRouter chat returned no choices", body=data)
    msg = choices[0].get("message") or {}

    for img in msg.get("images") or []:
        url = (
            (img.get("image_url") or {}).get("url")
            if isinstance(img, dict)
            else None
        )
        if not url and isinstance(img, dict):
            url = img.get("url") or img.get("b64_json")
        if isinstance(url, str) and url:
            if url.startswith("data:") or re.fullmatch(r"[A-Za-z0-9+/=\s]+", url[:80]):
                try:
                    return _decode_b64_image(url)
                except Exception:
                    pass
            if url.startswith("http"):
                with httpx.Client(timeout=60) as client:
                    r = client.get(url)
                    r.raise_for_status()
                    return r.content

    content = msg.get("content")
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("image_url", "output_image", "image"):
                url = (part.get("image_url") or part.get("image") or {}).get("url")
                if not url:
                    url = part.get("url") or part.get("b64_json")
                if isinstance(url, str) and url:
                    if url.startswith("http"):
                        with httpx.Client(timeout=60) as client:
                            r = client.get(url)
                            r.raise_for_status()
                            return r.content
                    return _decode_b64_image(url)
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                m = re.search(
                    r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", part["text"]
                )
                if m:
                    return base64.b64decode(m.group(1))

    if isinstance(content, str) and "base64," in content:
        m = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\s]+)", content)
        if m:
            return base64.b64decode(m.group(1))

    raise OpenRouterError("OpenRouter chat response missing image", body=data)


def _post_json(
    path: str,
    payload: dict[str, Any],
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    if not config.OPENROUTER_API_KEY:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured")
    url = f"{config.OPENROUTER_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    with httpx.Client(timeout=timeout or config.OPENROUTER_TIMEOUT_SEC) as client:
        resp = client.post(url, json=payload, headers=_headers())
    if resp.status_code >= 400:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text[:500]
        log.error("OpenRouter error %s %s: %s", path, resp.status_code, err_body)
        raise OpenRouterError(
            f"OpenRouter request failed ({resp.status_code})",
            status=resp.status_code,
            body=err_body,
        )
    return resp.json()


def edit_selfie(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> bytes:
    payload = build_edit_payload(image_bytes, mime, model=model, prompt=prompt)
    data = _post_json("images", payload)
    return _extract_image_bytes_from_images_api(data)


def _riverflow_image_config() -> dict[str, Any]:
    """Native Riverflow v2.5 background + scoring (OpenRouter image_config)."""
    bg_mode = config.RIVERFLOW_BG_MODE or "solid"
    if bg_mode not in ("solid", "transparent", "original"):
        bg_mode = "solid"
    cfg: dict[str, Any] = {
        "aspect_ratio": "3:4",
        "image_size": config.RIVERFLOW_IMAGE_SIZE or "1K",
        "background_mode": bg_mode,
        "scoring_prompt": GOSUSLUGI_SCORING_PROMPT,
        "scoring_rubric": GOSUSLUGI_SCORING_RUBRIC,
    }
    if bg_mode == "solid":
        hex_color = config.RIVERFLOW_BG_HEX or "#FFFFFF"
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", hex_color):
            hex_color = "#FFFFFF"
        cfg["background_hex_color"] = hex_color
    return cfg


def _is_riverflow_model(model: str | None = None) -> bool:
    return "riverflow" in (model or config.RIVERFLOW_MODEL or "").lower()


def build_riverflow_images_payload(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """POST /images payload for Riverflow (fallback path)."""
    model = model or config.RIVERFLOW_MODEL
    bg_mode = config.RIVERFLOW_BG_MODE or "solid"
    if bg_mode not in ("solid", "transparent", "original"):
        bg_mode = "solid"
    out_fmt = "png" if bg_mode == "transparent" else "jpeg"
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt or GOSUSLUGI_EDIT_PROMPT,
        "aspect_ratio": "3:4",
        "resolution": config.RIVERFLOW_IMAGE_SIZE or "1K",
        "output_format": out_fmt,
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        ],
        "image_config": _riverflow_image_config(),
    }
    effort = (config.RIVERFLOW_REASONING or "medium").strip().lower()
    if effort and effort != "none":
        payload["reasoning"] = {"effort": effort}
    return payload


def build_generic_edit_images_payload(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """POST /images for non-Riverflow editors (e.g. FLUX.2 Pro) — no image_config."""
    model = model or config.RIVERFLOW_MODEL
    return {
        "model": model,
        "prompt": prompt or GOSUSLUGI_EDIT_PROMPT,
        "aspect_ratio": _aspect_ratio_for_model(model),
        "output_format": "jpeg",
        "input_references": [
            {
                "type": "image_url",
                "image_url": {"url": _data_url(image_bytes, mime)},
            }
        ],
    }


def edit_selfie_riverflow(
    image_bytes: bytes,
    mime: str = "image/jpeg",
    *,
    model: str | None = None,
    prompt: str | None = None,
) -> bytes:
    """Gosuslugi white-bg via OpenRouter /images.

    Riverflow models get native background_mode + scoring; other models
    (e.g. google/gemini-2.5-flash-image) use a plain edit payload.
    """
    use_model = model or config.RIVERFLOW_MODEL
    use_prompt = prompt or GOSUSLUGI_EDIT_PROMPT
    if _is_riverflow_model(use_model) and use_prompt == GOSUSLUGI_EDIT_PROMPT:
        payload = build_riverflow_images_payload(
            image_bytes, mime, model=use_model, prompt=use_prompt
        )
    else:
        payload = build_generic_edit_images_payload(
            image_bytes, mime, model=use_model, prompt=use_prompt
        )
    data = _post_json(
        "images",
        payload,
        timeout=config.RIVERFLOW_TIMEOUT_SEC,
    )
    return _extract_image_bytes_from_images_api(data)


def edit_selfie_resume(
    image_bytes: bytes,
    mime: str = "image/jpeg",
) -> bytes:
    """Business-suit resume portrait via OpenRouter /images (no Gosuslugi scoring)."""
    payload = build_generic_edit_images_payload(
        image_bytes,
        mime,
        prompt=RESUME_SUIT_PROMPT,
    )
    data = _post_json(
        "images",
        payload,
        timeout=config.RIVERFLOW_TIMEOUT_SEC,
    )
    return _extract_image_bytes_from_images_api(data)
