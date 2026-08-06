import json
from pathlib import Path

from app import config
from app.pairs import save_pair


def test_save_pair_writes_in_out_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path)
    monkeypatch.setattr(config, "PAIRS_ENABLED", True)
    folder = save_pair(
        b"\xff\xd8\xffin",
        b"\xff\xd8\xffout",
        filename="selfie.jpg",
        meta={"model": "test"},
    )
    assert folder is not None
    assert (folder / "in.jpg").read_bytes() == b"\xff\xd8\xffin"
    assert (folder / "out.jpg").read_bytes() == b"\xff\xd8\xffout"
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "test"
    assert meta["in"] == "in.jpg"
    assert meta["out"] == "out.jpg"


def test_save_pair_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PAIRS_DIR", tmp_path)
    monkeypatch.setattr(config, "PAIRS_ENABLED", False)
    assert save_pair(b"a", b"b") is None
    assert list(tmp_path.iterdir()) == []
