"""BIA-02.2: testes sobre corpus artificial, sem fontes científicas reais."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from apps.api.main import make_app
from bia_core.db import (Annotation, Document, Passage, PassageReview,
                         ScientificRelation, RelationModelRun, RelationNegativeAssessment, make_session_factory, now)
from bia_core.relation_learning import (BiaRelationClassifier, RelationConfig,
                                         evidence_window, grouped_splits, predict_candidates,
                                         reviewed_relations, train_relations)
from bia_core.relations import certification_key

KEYS = {"admin": {"actor": "admin", "role": "admin"},
        "curator": {"actor": "curador", "role": "curator"},
        "viewer": {"actor": "visualizador", "role": "viewer"}}


def headers(role):
    return {"Authorization": "Bearer " + role}


def add_relation(session, index: int, decision: str, *, allowed: bool = True,
                 repeated_text: str | None = None, stale: bool = False, reviewed: bool = True,
                 explicit: bool = True):
    text = repeated_text or f"Material{index} apresenta Propriedade{index} no exemplo artificial."
    document = Document(title="Ficção", doi=None, year=None, rights_basis="open_access",
                        training_allowed=allowed, asset_path="nao-e-arquivo-real",
                        sha256=sha256((str(index) + text).encode()).hexdigest(), status="indexed")
    session.add(document)
    session.flush()
    passage = Passage(document_id=document.id, page=1, text=text, start_offset=0, end_offset=len(text))
    session.add(passage)
    session.flush()
    a_start = text.index("Material")
    b_start = text.index("Propriedade")
    a = Annotation(passage_id=passage.id, label="MATERIAL", start=a_start,
                   end=a_start + len(f"Material{index}") if repeated_text is None else a_start + len("Material0"),
                   actor="autor", status="approved", reviewer="colega", reviewed_at=now())
    b = Annotation(passage_id=passage.id, label="PROPERTY", start=b_start,
                   end=b_start + len(f"Propriedade{index}") if repeated_text is None else b_start + len("Propriedade0"),
                   actor="autor", status="approved", reviewer="colega", reviewed_at=now())
    session.add_all([a, b])
    session.flush()
    review = PassageReview(passage_id=passage.id, status="certified",
                           submitted_by="autor", certified_by="colega", submitted_at=now(), certified_at=now())
    session.add(review)
    session.flush()
    rel = ScientificRelation(passage_id=passage.id, subject_annotation_id=a.id, object_annotation_id=b.id,
                             relation_type="MATERIAL_HAS_PROPERTY", evidence_start=0, evidence_end=len(text),
                             certification_key="f" * 64 if stale else certification_key(review),
                             status=decision, proposed_by="autor", reviewed_by="colega" if reviewed else None,
                             reviewed_at=now() if reviewed else None)
    session.add(rel)
    session.flush()
    if decision == "rejected" and explicit and reviewed:
        session.add(RelationNegativeAssessment(relation_id=rel.id, assessed_by="colega", reason="relation_absent"))
    return passage, a, b, rel


@pytest.fixture
def sf(tmp_path):
    return make_session_factory("sqlite:///" + str(tmp_path / "curadoria.sqlite"))


def corpus(sf, n=7):
    with sf() as session:
        for index in range(n):
            add_relation(session, index, "approved" if index % 2 else "rejected")
        session.commit()


def test_corpus_only_explicit_current_independently_reviewed(sf):
    with sf() as session:
        add_relation(session, 0, "approved")
        add_relation(session, 1, "rejected")
        add_relation(session, 2, "approved", allowed=False)
        add_relation(session, 3, "approved", stale=True)
        add_relation(session, 4, "rejected", reviewed=False)
        add_relation(session, 5, "rejected", explicit=False)
        session.commit()
        records = reviewed_relations(session)
        assert sorted(r["label"] for r in records) == [0, 1]
        assert {r["type"] for r in records} == {"MATERIAL_HAS_PROPERTY"}
        assert all(r["reviewed_by"] != r["proposed_by"] for r in records)


def test_reopening_passage_excludes_stale_and_inference_candidates(sf):
    with sf() as session:
        passage, _, _, rel = add_relation(session, 0, "approved")
        session.commit()
        assert len(reviewed_relations(session)) == 1
        review = session.get(PassageReview, passage.id)
        review.status = "draft"
        review.certified_at = None
        session.commit()
        assert reviewed_relations(session) == []
        assert session.get(ScientificRelation, rel.id).status == "approved"  # histórico mantido


def test_grouped_split_avoids_duplicate_passages():
    rows = [
        {"document_sha256": "a", "passage_sha256": "same", "label": 1},
        {"document_sha256": "b", "passage_sha256": "same", "label": 0},
        {"document_sha256": "c", "passage_sha256": "unique1", "label": 1},
        {"document_sha256": "d", "passage_sha256": "unique2", "label": 0},
        {"document_sha256": "e", "passage_sha256": "unique3", "label": 1},
    ]
    one = grouped_splits(rows, 17)
    assert one == grouped_splits(rows, 17)
    assert one["a"] == one["b"]
    assert set(one.values()) <= {"train", "val", "test"}


def test_evidence_window_keeps_arguments_and_fails_closed():
    row = {"text": "X" * 70 + "AB" + "Y" * 60 + "CD" + "Z" * 70,
           "subject": (70, 72), "object": (132, 134), "evidence": (50, 150)}
    window, subj, obj = evidence_window(row, 70)
    assert window[subj[0]:subj[1]] == "AB"
    assert window[obj[0]:obj[1]] == "CD"
    with pytest.raises(ValueError, match="excede a janela"):
        evidence_window(row, 30)
    bad = {**row, "evidence": (100, 150)}
    with pytest.raises(ValueError, match="não contém"):
        evidence_window(bad, 70)


def test_training_requires_positive_and_explicit_negative(sf, tmp_path):
    with sf() as session:
        add_relation(session, 0, "approved")
        session.commit()
        with pytest.raises(ValueError, match="positivos e negativos"):
            train_relations(session, tmp_path / "invalid", epochs=1)
    assert not (tmp_path / "invalid").exists()


def test_neural_training_checkpoint_and_ranking(sf, tmp_path):
    corpus(sf)
    folder = tmp_path / "relation-model"
    with sf() as session:
        metrics = train_relations(session, folder, epochs=1,
            config_override={"dim": 16, "heads": 4, "layers": 1, "max_length": 96})
        assert metrics["train_positive"] >= 1 and metrics["train_negative_explicit"] >= 1
        assert metrics["total_reviewed"] == 7 and metrics["scientifically_validated"] is False
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["architecture"] == "bia-relation-char-transformer"
        assert len(manifest["dataset_sha256"]) == 64
        assert not any(x == "scientifically_validated" and y is True for x, y in metrics.items())
        row = reviewed_relations(session)[0]
        passage = session.get(Passage, row["passage_id"])
        a = session.get(Annotation, session.get(ScientificRelation, row["relation_id"]).subject_annotation_id)
        b = session.get(Annotation, session.get(ScientificRelation, row["relation_id"]).object_annotation_id)
        candidate = {"subject_annotation_id": a.id, "object_annotation_id": b.id,
                     "relation_type": row["type"], "subject": row["subject"], "object": row["object"],
                     "evidence_start": 0, "evidence_end": len(passage.text)}
        result = predict_candidates(passage, [candidate], folder)
        assert len(result) == 1 and result[0]["status"] == "candidate_unverified"
        assert result[0]["scientifically_validated"] is False
        assert 0 <= result[0]["model_score_uncalibrated"] <= 1
        assert len(result[0]["evidence_quote"]) == len(passage.text)
        (folder / "tokenizer.json").write_text("{}")
        with pytest.raises(ValueError, match="Integridade"):
            predict_candidates(passage, [candidate], folder)


def test_model_forward_respects_direction():
    import torch
    conf = RelationConfig(vocab_size=6, dim=16, max_length=12)
    model = BiaRelationClassifier(conf)
    inp = torch.tensor([[2, 3, 4, 5]])
    forward = model(inp, (0, 1), (2, 3), 0)
    backward = model(inp, (2, 3), (0, 1), 0)
    assert forward.shape == (1, 2) and not torch.equal(forward, backward)
    with pytest.raises(ValueError, match="Offsets"):
        model(inp, (0, 8), (0, 1), 0)


def test_api_model_job_and_restricted_predictions(sf, tmp_path):
    corpus(sf)
    url = str(sf.kw["bind"].url)
    app = make_app(db_url=url, keys=KEYS, storage_dir=tmp_path / "docs", artifact_root=tmp_path / "models")
    client = TestClient(app)
    assert client.post("/v1/relation-models/train", headers=headers("viewer"), json={"epochs": 1}).status_code == 403
    response = client.post("/v1/relation-models/train", headers=headers("admin"), json={"epochs": 1})
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    for _ in range(100):
        status = client.get(f"/v1/relation-models/runs/{run_id}", headers=headers("curator"))
        assert status.status_code == 200
        if status.json()["status"] in ("completed", "failed"):
            break
        time.sleep(.05)
    assert status.json()["status"] == "completed", status.json()
    with sf() as session:
        passage = session.scalar(select(Passage).order_by(Passage.id))
        pid = passage.id
    assert client.get(f"/v1/passages/{pid}/relation-predictions", headers=headers("viewer")).status_code == 403
    ranked = client.get(f"/v1/passages/{pid}/relation-predictions", headers=headers("curator"))
    assert ranked.status_code == 200, ranked.text
    content = ranked.json()
    assert content["status"] == "experimental_unverified"
    assert content["scientifically_validated"] is False and content["candidates"]
    assert content["candidates"][0]["status"] == "candidate_unverified"
    with sf() as session:
        # Não há transição de status, nem novas relações criadas pela inferência.
        assert session.scalar(select(RelationModelRun).where(RelationModelRun.id == run_id)) is not None
        assert session.query(ScientificRelation).count() == 7
    with sf() as session:
        assert session.get(RelationModelRun, run_id).status == "completed"


def test_negative_training_label_requires_explicit_review_api(sf, tmp_path):
    with sf() as session:
        a = add_relation(session, 0, "pending")[3]
        b = add_relation(session, 1, "pending")[3]
        c = add_relation(session, 2, "pending")[3]
        session.commit()
        ids = a.id, b.id, c.id
    app = make_app(db_url=str(sf.kw["bind"].url), keys=KEYS,
                   storage_dir=tmp_path / "docs", artifact_root=tmp_path / "models")
    client = TestClient(app)
    forbidden = client.post(f"/v1/relations/{ids[0]}/review", headers=headers("curator"),
        json={"decision": "approved", "explicit_no_relation": True})
    assert forbidden.status_code == 422
    ordinary = client.post(f"/v1/relations/{ids[0]}/review", headers=headers("curator"),
        json={"decision": "rejected"})
    assert ordinary.status_code == 200
    confirmed = client.post(f"/v1/relations/{ids[1]}/review", headers=headers("curator"),
        json={"decision": "rejected", "explicit_no_relation": True})
    assert confirmed.status_code == 200
    approved = client.post(f"/v1/relations/{ids[2]}/review", headers=headers("curator"),
        json={"decision": "approved"})
    assert approved.status_code == 200
    with sf() as session:
        assert session.get(RelationNegativeAssessment, ids[0]) is None
        assert session.get(RelationNegativeAssessment, ids[1]).reason == "relation_absent"
        data = reviewed_relations(session)
        assert sorted(r["label"] for r in data) == [0, 1]
