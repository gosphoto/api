from app.openrouter import EDIT_PROMPT, build_edit_payload


def test_prompt_mentions_no_fringe():
    lower = EDIT_PROMPT.lower()
    assert "fringe" in lower or "spill" in lower or "halo" in lower


def test_prompt_forbids_skin_smoothing():
    lower = EDIT_PROMPT.lower()
    assert "smooth" in lower or "beauty" in lower
    assert "pore" in lower or "face" in lower


def test_prompt_allows_only_shoulders_and_bg():
    lower = EDIT_PROMPT.lower()
    assert "shoulder" in lower
    assert "forbidden" in lower
    assert "head" in lower
    assert "identity" in lower


def test_prompt_forbids_invented_jewelry():
    lower = EDIT_PROMPT.lower()
    assert "earring" in lower
    assert "nose ring" in lower or "piercing" in lower
    from app.openrouter import GOSUSLUGI_EDIT_PROMPT

    ru = GOSUSLUGI_EDIT_PROMPT
    assert "серьг" in ru
    assert "не добавляй" in ru


def test_prompt_allows_light_rescue_for_dark_source():
    from app.openrouter import GOSUSLUGI_EDIT_PROMPT

    ru = GOSUSLUGI_EDIT_PROMPT
    assert "тёмн" in ru
    assert "некачественн" in ru
    assert "небольш" in ru and "ретуш" in ru
    assert "Не омолаживай" in ru
    lower = EDIT_PROMPT.lower()
    assert "underexposed" in lower or "dark" in lower
    assert "beauty" in lower



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
