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


def test_gosuslugi_prompt_is_resume_with_white_bg_and_spot_cleanup():
    from app.openrouter import GOSUSLUGI_EDIT_PROMPT, RESUME_SUIT_PROMPT

    assert GOSUSLUGI_EDIT_PROMPT != RESUME_SUIT_PROMPT
    g = GOSUSLUGI_EDIT_PROMPT.lower()
    assert "business suit" in g
    assert "blazer" in g
    assert "child" in g
    assert "t-shirt" in g or "tank" in g
    assert "navy" in g or "charcoal" in g or "pale" in g
    assert "pink" in g
    assert "girl" in g
    assert "boy" in g
    assert "розовая майка" in GOSUSLUGI_EDIT_PROMPT
    assert "no age change" in g
    assert "#ffffff" in g
    assert "spot" in g or "blemish" in g
    assert "level shoulders" in g or "upright" in g
    assert "glare" in g or "reflection" in g
    assert "gray" in g or "grey" in g  # forbidden gray backdrop called out
    assert "overexpose" in g or "bleach" in g
    assert "soft studio lighting" not in g
    assert "even skin tone" not in g
    assert "neutral" in g and "smile" in g
    assert "flyaway" in g or "wisp" in g
    assert "erase" in g or "remove" in g
    assert "must" in g
    assert "halo" in g
    assert "zero stray" in g or "invisible" in g
    assert "leftover" in g or "original-wall" in g or "wall" in g
    assert "do not restyle" in g
    assert "do not tuck" in g
    assert "behind the ears" in g
    assert "sunglasses" in g
    assert "headwear" in g or "hat" in g
    assert "hijab" in g or "religious" in g
    assert "religious clothing" in g
    assert "keep that clothing" in g
    assert "do not replace" in g
    assert "abaya" in g
    assert "recolor" in g
    assert "white hijab" in g
    assert "gray" in g or "grey" in g
    assert "collar" in g
    assert "blazer" in g or "lapel" in g
    assert "light-blue" in g or "light blue" in g or "голубая" in g
    assert "either (1)" not in g
    assert "do both" in g
    assert "and (2)" in g
    assert "never one without the other" in g
    r = RESUME_SUIT_PROMPT.lower()
    assert "religious clothing" in r
    assert "keep that clothing" in r
    assert "do not replace" in r
    assert "toy" in g or "foreign object" in g
    assert "mole" in g or "freckle" in g
    assert "invent" in g
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


def test_riverflow_payload_uses_passed_model():
    from app.openrouter import build_riverflow_images_payload

    payload = build_riverflow_images_payload(
        b"fake", "image/jpeg", model="sourceful/riverflow-v2.5-pro"
    )
    assert payload["model"] == "sourceful/riverflow-v2.5-pro"
    assert payload["image_config"]["background_mode"] == "solid"
