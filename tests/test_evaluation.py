"""BIA-02.3: cenários sintéticos, nenhum dado científico real incluído."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select
import pytest

from apps.api.main import make_app
from bia_core.db import RelationModelRun, make_session_factory
from bia_core.evaluation import build_evaluation_report, summarise_predictions, verified_report
from bia_core.relation_learning import grouped_splits, publication_hash, train_relations
from test_relation_learning import KEYS, headers, add_relation


def test_duplicate_doi_normalization_and_splits():
    same = publication_hash(" https://doi.org/10.1000/ABC ")
    assert same == publication_hash("DOI:10.1000/abc")
    assert publication_hash(" ") is None
    docs = [
        {"document_sha256": "doc-a", "publication_sha256": same, "passage_sha256": "a"},
        {"document_sha256": "doc-b", "publication_sha256": same, "passage_sha256": "b"},
        {"document_sha256": "doc-c", "publication_sha256": None, "passage_sha256": "same"},
        {"document_sha256": "doc-d", "publication_sha256": None, "passage_sha256": "same"},
        {"document_sha256": "doc-e", "publication_sha256": None, "passage_sha256": "e"},
        {"document_sha256": "doc-f", "publication_sha256": None, "passage_sha256": "f"},
    ]
    for index, doc in enumerate(docs):
        doc["label"] = index % 2
    for seed in range(20):
        result = grouped_splits(docs, seed)
        assert result["doc-a"] == result["doc-b"]
        assert result["doc-c"] == result["doc-d"]
        assert result == grouped_splits(docs, seed)


def test_empty_support_metrics_are_null_not_zero():
    metrics = summarise_predictions([], ("A", "B"))
    assert metrics["overall"]["examples"] == 0
    assert metrics["overall"]["f1"] is None
    assert metrics["overall"]["precision"] is None
    assert metrics["overall"]["macro_f1_defined_types"] is None
    assert metrics["by_type"]["A"]["f1"] is None
    with pytest.raises(ValueError):
        summarise_predictions([{"label": 2, "prediction": 0, "relation_type": "A"}], ("A",))


def test_evaluation_counts_errors_and_document_breakdown():
    entries = [
        {"split": "test", "relation_id": "a", "document_sha256": "doc-a", "passage_sha256": "aa", "relation_type": "A", "label": 1, "prediction": 0, "score_uncalibrated": .2},
        {"split": "test", "relation_id": "b", "document_sha256": "doc-b", "passage_sha256": "bb", "relation_type": "A", "label": 0, "prediction": 1, "score_uncalibrated": .7},
        {"split": "val", "relation_id": "c", "document_sha256": "doc-c", "passage_sha256": "cc", "relation_type": "B", "label": 1, "prediction": 1, "score_uncalibrated": .6},
    ]
    assignments = {"doc-a": "test", "doc-b": "test", "doc-c": "val", "doc-d": "train"}
    result = build_evaluation_report(entries, relation_types=("A", "B"), assignments=assignments,
                                     dataset_sha256="f" * 64)
    assert result["test"]["overall"]["fn"] == 1
    assert result["test"]["overall"]["fp"] == 1
    assert result["test"]["document_count"] == 2
    assert {item["kind"] for item in result["test"]["errors"]} == {"false_positive", "false_negative"}
    assert result["validation"]["overall"]["f1"] == 1.0
    assert result["scientifically_validated"] is False
    assert "text" not in json.dumps(result)
    bad = {"doc-a": "test", "doc-b": "test", "doc-c": "val", "doc-d": "train"}
    bad["doc-a"] = "train"  # records inconsistent with assignments: must be rejected
    with pytest.raises(ValueError, match="inconsistente"):
        build_evaluation_report(entries, relation_types=("A", "B"), assignments=bad,
                                dataset_sha256="f" * 64)


def test_end_to_end_holdout_integrity_and_admin_api(tmp_path):
    url = "sqlite:///" + str(tmp_path / "bia.sqlite")
    sf = make_session_factory(url)
    with sf() as session:
        for i in range(10):
            add_relation(session, i, "approved" if i % 2 else "rejected")
        session.commit()
    folder = tmp_path / "models" / "relations" / "run123"
    with sf() as session:
        metrics = train_relations(session, folder, epochs=1, seed=10,
                                 config_override={"dim": 16, "heads": 4, "layers": 1, "max_length": 96})
        assert metrics["test_examples"] > 0
        assert metrics["validation_examples"] > 0
        assert metrics["holdout_evaluable"] is True
        run = RelationModelRun(id="run123", status="completed", artifact_dir=str(folder), metrics_json=json.dumps(metrics))
        session.add(run)
        session.commit()
    report = verified_report(folder)
    assert report["test"]["evaluable"] is True
    assert report["documents"]["train"] + report["documents"]["validation"] + report["documents"]["test"] == 10
    snapshot = json.loads((folder / "dataset_snapshot.json").read_text())
    docs = {}
    for record in snapshot:
        assert docs.setdefault(record["document_sha256"], record["split"]) == record["split"]
    app = make_app(db_url=url, storage_dir=tmp_path / "docs", artifact_root=tmp_path / "models", keys=KEYS)
    client = TestClient(app)
    path = "/v1/relation-models/runs/run123/evaluation"
    assert client.get(path, headers=headers("viewer")).status_code == 403
    assert client.get(path, headers=headers("curator")).status_code == 403
    assert client.get(path, headers=headers("admin")).status_code == 200
    (folder / "evaluation_report.json").write_text('{"tampered":true}', encoding="utf-8")
    assert client.get(path, headers=headers("admin")).status_code == 409
    with pytest.raises(ValueError, match="Integridade"):
        verified_report(folder)


def test_holdout_absent_is_unmeasurable():
    result = build_evaluation_report([], relation_types=("A",), assignments={"doc-1": "train"}, dataset_sha256="0"*64)
    assert result["test"]["evaluable"] is False
    assert result["test"]["overall"]["f1"] is None
    assert result["documents"]["test"] == 0
