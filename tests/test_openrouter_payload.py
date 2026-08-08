from app.openrouter import EDIT_PROMPT, GOSUSLUGI_NANO_PROMPT, build_edit_payload


def test_edit_prompt_keeps_identity_and_blocks_hair_holes():
    lower = EDIT_PROMPT.lower()
    assert "identity" in lower or "original" in lower
    assert "hair" in lower
    assert "white holes" in lower or "swiss-cheese" in lower
    assert "shoulder" in lower or "clothing" in lower


def test_gosuslugi_nano_prompt_blocks_holes_inside_hair():
    assert "ВНУТРИ массы волос" in GOSUSLUGI_NANO_PROMPT
    assert "#FFFFFF" in GOSUSLUGI_NANO_PROMPT
    assert "ушам" in GOSUSLUGI_NANO_PROMPT or "ушей" in GOSUSLUGI_NANO_PROMPT


def test_payload_uses_transparent_png_when_enabled(monkeypatch):
    monkeypatch.setattr("app.openrouter.config.OPENROUTER_IMAGE_MODEL", "openai/gpt-image-1")
    monkeypatch.setattr("app.openrouter.config.OPENROUTER_TRANSPARENT_BG", True)
    payload = build_edit_payload(b"fake", "image/jpeg")
    assert payload["model"] == "openai/gpt-image-1"
    assert payload["output_format"] == "png"
    assert payload["background"] == "transparent"
    assert payload["aspect_ratio"] == "2:3"
    assert payload["input_references"]


def test_payload_omits_transparent_when_disabled(monkeypatch):
    monkeypatch.setattr("app.openrouter.config.OPENROUTER_TRANSPARENT_BG", False)
    payload = build_edit_payload(b"fake", "image/jpeg")
    assert "background" not in payload
    assert payload["output_format"] == "jpeg"


def test_gemini_keeps_3_4_aspect(monkeypatch):
    monkeypatch.setattr(
        "app.openrouter.config.OPENROUTER_IMAGE_MODEL", "google/gemini-2.5-flash-image"
    )
    monkeypatch.setattr("app.openrouter.config.OPENROUTER_TRANSPARENT_BG", False)
    payload = build_edit_payload(b"fake", "image/jpeg")
    assert payload["aspect_ratio"] == "3:4"
