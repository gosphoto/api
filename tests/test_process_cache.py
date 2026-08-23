import json
from pathlib import Path

from PIL import Image
import io

from app import config
from app.process_cache import content_hash, lookup, put
from app.results import clone_result, load_file, load_meta, save_result


def _jpeg(color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (40, 50), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_content_hash_stable():
    data = b"same-bytes"
    assert content_hash(data) == content_hash(data)
    assert content_hash(b"other") != content_hash(data)


def test_clone_result_copies_files_and_resets_payment(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    dig = _jpeg()
    prn = _jpeg((40, 50, 60))
    source_id = save_result(
        dig,
        prn,
        meta={"doc_type": "passport_rf", "paid": True, "payment_id": "x"},
    )
    assert source_id is not None
    cloned_id = clone_result(source_id)
    assert cloned_id is not None
    assert cloned_id != source_id
    meta = load_meta(cloned_id)
    assert meta is not None
    assert meta["paid"] is False
    assert meta["cloned_from"] == source_id
    assert "payment_id" not in meta
    assert load_file(cloned_id, "digital.jpg") == dig
    assert load_file(cloned_id, "print.jpg") == prn


def test_process_cache_put_and_lookup(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROCESS_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_VERSION", "1")

    data = b"oriented-upload-bytes"
    dig = _jpeg()
    prn = _jpeg((1, 2, 3))
    source_id = save_result(
        dig,
        prn,
        meta={
            "doc_type": "passport_rf",
            "doc_label": "Паспорт РФ",
            "pipeline": ["gate", "riverflow", "crop", "print_10x15"],
            "edit": {"model": "test-model"},
            "compliance": {"pass": True},
            "print_sheet": {"copies": 4},
        },
    )
    assert source_id is not None

    put(data, "passport_rf", source_id)
    cache_files = list((tmp_path / "cache").rglob("*.json"))
    assert len(cache_files) == 1
    entry = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert entry["result_id"] == source_id

    hit_id = lookup(data, "passport_rf")
    assert hit_id is not None
    assert hit_id != source_id
    assert load_file(hit_id, "digital.jpg") == dig

    assert lookup(data, "zagran") is None
    assert lookup(b"other-bytes", "passport_rf") is None


def test_process_cache_put_is_first_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROCESS_CACHE_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_VERSION", "1")

    data = b"same-photo"
    first_id = save_result(_jpeg(), _jpeg((1, 1, 1)))
    second_id = save_result(_jpeg((2, 2, 2)), _jpeg((3, 3, 3)))
    assert first_id and second_id

    put(data, "passport_rf", first_id)
    put(data, "passport_rf", second_id)

    cache_path = next((tmp_path / "cache").rglob("*.json"))
    entry = json.loads(cache_path.read_text(encoding="utf-8"))
    assert entry["result_id"] == first_id


def test_process_cache_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "RESULTS_ENABLED", True)
    monkeypatch.setattr(config, "PROCESS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROCESS_CACHE_ENABLED", False)

    rid = save_result(_jpeg(), _jpeg((5, 5, 5)))
    put(b"x", "passport_rf", rid)
    assert not (tmp_path / "cache").exists()
    assert lookup(b"x", "passport_rf") is None
