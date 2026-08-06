import json
from pathlib import Path

from app import config
from app.rejecteds import save_rejected


def test_save_rejected_writes_image_and_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REJECTEDS_DIR", tmp_path)
    monkeypatch.setattr(config, "REJECTEDS_ENABLED", True)
    data = b"\xff\xd8\xfffakejpeg"
    path = save_rejected(
        data,
        reason="no_face",
        message="no face",
        metrics={"face_count": 0},
        filename="bad selfie!.jpg",
    )
    assert path is not None
    assert path.is_file()
    assert path.read_bytes() == data
    assert "no_face" in path.name
    meta_path = Path(str(path) + ".json")
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["reason"] == "no_face"
    assert meta["bytes"] == len(data)


def test_save_rejected_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REJECTEDS_DIR", tmp_path)
    monkeypatch.setattr(config, "REJECTEDS_ENABLED", False)
    assert save_rejected(b"abc", reason="blur") is None
    assert list(tmp_path.iterdir()) == []
