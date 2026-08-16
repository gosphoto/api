import json
from pathlib import Path

from app import doc_metrics


def test_record_doc_type_appends_jsonl(tmp_path, monkeypatch):
    monkeypatch.setattr(doc_metrics.config, "RESULTS_DIR", tmp_path)
    doc_metrics.record_doc_type(
        doc_type="zagran", result_id="a" * 32, dpi=300, source="process"
    )
    doc_metrics.record_doc_type(
        doc_type="passport_rf", result_id="b" * 32, dpi=600, source="process"
    )
    path = tmp_path / "_metrics" / "doc_type.jsonl"
    assert path.is_file()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == 2
    assert rows[0]["doc_type"] == "zagran"
    assert rows[0]["dpi"] == 300
    assert rows[1]["doc_type"] == "passport_rf"
