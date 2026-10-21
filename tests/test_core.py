from __future__ import annotations

from io import BytesIO
import json

import pytest
from reportlab.pdfgen import canvas
from sqlalchemy import select
from fastapi.testclient import TestClient

from apps.api.main import make_app
from bia_core.db import Annotation, Document, Passage, make_session_factory
from bia_core.retrieval import bm25, tokenize
from bia_core.tokenizer import CharTokenizer
from bia_core.training import build_samples, document_splits, extract, source_dataset, train, _spans
from bia_core.model import LABELS


KEYS = {
    "editor-secret": {"actor": "editor", "role": "curator", "institutes": ["inovaligno"]},
    "reviewer-secret": {"actor": "reviewer", "role": "curator", "institutes": ["inovaligno"]},
    "admin-secret": {"actor": "admin", "role": "admin"},
    "viewer-secret": {"actor": "reader", "role": "viewer"},
}


def bearer(key: str):
    return {"Authorization": f"Bearer {key}"}


def pdf(text: str):
    buffer = BytesIO()
    document = canvas.Canvas(buffer, invariant=1)
    document.drawString(60, 700, text)
    document.save()
    return buffer.getvalue()


@pytest.fixture()
def env(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.sqlite3'}"
    app = make_app(db_url=url, storage_dir=tmp_path / 'docs', artifact_root=tmp_path / 'weights', keys=KEYS)
    return TestClient(app), make_session_factory(url), tmp_path


def add_pdf(client, *, key="editor-secret", text="Celulose possui resistencia de 45 MPa", rights="open_access", training="true"):
    return client.post("/v1/documents", headers=bearer(key),
                       files={"file": ("artigo.pdf", pdf(text), "application/pdf")},
                       data={"title": "Artigo de demonstração", "rights_basis": rights, "training_allowed": training,
                             **({"institute": "inovaligno"} if rights == "institutional_permission" else {})})


def test_tokenizer_round_trip_unicode():
    source = "Celulose μm 30 °C, polímero β"
    codec = CharTokenizer.fit([source])
    assert codec.decode(codec.encode(source)) == source
    assert codec.vocab["<pad>"] == 0


def test_bm25_returns_grounded_rank():
    score = bm25("celulose", ["celulose celulose propriedades", "óleo vegetal", "celulose com lignina"])
    assert score[0] > score[1] == 0
    assert all(value >= 0 for value in score)
    assert tokenize("µm e 90°C")


def test_auth_ingestion_search_and_visibility(env):
    client, _, _ = env
    assert client.get("/v1/search?q=celulose").status_code == 401
    assert add_pdf(client, key="viewer-secret").status_code == 403
    added = add_pdf(client)
    assert added.status_code == 201, added.text
    document_id = added.json()["id"]
    assert add_pdf(client).status_code == 409
    assert client.get(f"/v1/documents/{document_id}/passages", headers=bearer("viewer-secret")).json()[0]["page"] == 1
    results = client.get("/v1/search?q=celulose", headers=bearer("viewer-secret")).json()
    assert results["status"] == "ok"
    assert results["results"][0]["page"] == 1
    assert results["results"][0]["document_id"] == document_id
    assert results["results"][0]["doi"] is None

    restricted = add_pdf(client, text="Xilana especial", rights="institutional_permission")
    assert restricted.status_code == 201
    private_id = restricted.json()["id"]
    assert client.get(f"/v1/documents/{private_id}", headers=bearer("viewer-secret")).status_code == 404
    assert client.get("/v1/search?q=xilana", headers=bearer("viewer-secret")).json()["status"] == "insufficient_evidence"
    assert client.get("/v1/search?q=xilana", headers=bearer("reviewer-secret")).json()["status"] == "ok"


def test_pdf_invalid_and_size(env, monkeypatch):
    client, _, _ = env
    invalid = client.post("/v1/documents", headers=bearer("editor-secret"),
                          files={"file": ("fake.pdf", b"not a PDF", "application/pdf")},
                          data={"title": "Falha", "rights_basis": "open_access"})
    assert invalid.status_code == 422
    monkeypatch.setenv("BIA_MAX_PDF_BYTES", "100")
    assert add_pdf(client).status_code == 413


def test_annotation_review_and_training_with_provenance(env):
    client, factory, tmp_path = env
    uploaded = add_pdf(client, text="Celulose 45 MPa")
    doc_id = uploaded.json()["id"]
    passage = client.get(f"/v1/documents/{doc_id}/passages", headers=bearer("editor-secret")).json()[0]
    start = passage["text"].index("Celulose")
    entity = client.post("/v1/annotations", headers=bearer("editor-secret"),
                         json={"passage_id": passage["id"], "label": "MATERIAL", "start": start, "end": start + 8})
    assert entity.status_code == 201, entity.text
    annotation_id = entity.json()["id"]
    same = client.post(f"/v1/annotations/{annotation_id}/review", headers=bearer("editor-secret"), json={"decision": "approved"})
    assert same.status_code == 403
    review = client.post(f"/v1/annotations/{annotation_id}/review", headers=bearer("reviewer-secret"), json={"decision": "approved"})
    assert review.status_code == 200
    overlap = client.post("/v1/annotations", headers=bearer("editor-secret"),
                          json={"passage_id": passage["id"], "label": "PROCESS", "start": start, "end": start + 3})
    assert overlap.status_code == 409
    with factory() as session:
        assert source_dataset(session) == []  # revisão isolada NÃO certifica texto completo
    assert client.post(f"/v1/passages/{passage['id']}/submit", headers=bearer("editor-secret")).status_code == 200
    assert client.post(f"/v1/passages/{passage['id']}/certify", headers=bearer("editor-secret")).status_code == 403
    assert client.post(f"/v1/passages/{passage['id']}/certify", headers=bearer("reviewer-secret")).status_code == 200
    with factory() as session:
        output = tmp_path / "run"
        metrics = train(session, output, epochs=2, seed=42, config_override={"dim": 16, "heads": 4, "layers": 1, "max_length": 64})
        assert metrics["train_examples"] > 0
        assert metrics["validation_entity_f1"] is None
        assert metrics["certified_passages"] == 1
        assert json.loads((output / "manifest.json").read_text())["dataset_sha256"]
        result = extract(passage["text"], output)
        assert result["scientifically_validated"] is False
        assert len(result["model_sha256"]) == 64
        for item in result["entities"]:
            assert item["quote"] == passage["text"][item["start"]:item["end"]]
        (output / "weights.pt").write_bytes(b"corrupted")
        with pytest.raises(ValueError, match="Checkpoint inválido"):
            extract(passage["text"], output)


def test_no_training_without_explicit_permission(env):
    client, factory, tmp_path = env
    uploaded = add_pdf(client, text="Lignina nova", training="false")
    pid = client.get(f"/v1/documents/{uploaded.json()['id']}/passages", headers=bearer("editor-secret")).json()[0]["id"]
    ann = client.post("/v1/annotations", headers=bearer("editor-secret"),
                      json={"passage_id": pid, "label": "MATERIAL", "start": 0, "end": 7}).json()["id"]
    client.post(f"/v1/annotations/{ann}/review", headers=bearer("reviewer-secret"), json={"decision": "approved"})
    assert client.post(f"/v1/passages/{pid}/submit", headers=bearer("editor-secret")).status_code == 200
    assert client.post(f"/v1/passages/{pid}/certify", headers=bearer("reviewer-secret")).status_code == 200
    with factory() as session:
        with pytest.raises(ValueError, match="Não há trechos certificados"):
            train(session, tmp_path / "should-not-exist")
    assert client.get(f"/v1/extract/{pid}", headers=bearer("editor-secret")).json()["status"] == "insufficient_evidence"


def test_splits_are_by_document_not_passage():
    items = [{"sha256": hex(i), "text": "x"} for i in range(12)]
    items += [{"sha256": "0x1", "text": "y"}]
    splits = document_splits(items)
    assert len(splits) == 12
    assert {"train", "val", "test"} == set(splits.values())
    assert splits["0x1"] == splits[items[1]["sha256"]]


def test_certification_workflow_blocks_incomplete_and_frozen_labels(env):
    client, factory, _ = env
    uploaded = add_pdf(client, text="Celulose 45 MPa, lignina")
    pid = client.get(f"/v1/documents/{uploaded.json()['id']}/passages", headers=bearer("editor-secret")).json()[0]["id"]
    annotation = client.post("/v1/annotations", headers=bearer("editor-secret"),
        json={"passage_id": pid, "label": "MATERIAL", "start": 0, "end": 8}).json()["id"]
    # Rascunho sem cobertura total não pode declarar o fundo como O.
    with factory() as session:
        assert not source_dataset(session)
    assert client.post(f"/v1/passages/{pid}/submit", headers=bearer("editor-secret")).status_code == 409
    assert client.post(f"/v1/annotations/{annotation}/review", headers=bearer("reviewer-secret"), json={"decision":"approved"}).status_code == 200
    assert client.post(f"/v1/passages/{pid}/submit", headers=bearer("editor-secret")).json()["status"] == "submitted"
    assert client.post(f"/v1/passages/{pid}/certify", headers=bearer("editor-secret")).status_code == 403
    assert client.post("/v1/annotations", headers=bearer("editor-secret"),
        json={"passage_id": pid, "label":"UNIT", "start": 12, "end": 15}).status_code == 409
    assert client.post(f"/v1/passages/{pid}/certify", headers=bearer("reviewer-secret")).json()["status"] == "certified"
    with factory() as session:
        rows = source_dataset(session)
        assert len(rows) == 1 and rows[0]["annotations"][0]["label"] == "MATERIAL"
    assert client.post(f"/v1/passages/{pid}/reopen", headers=bearer("editor-secret")).json()["status"] == "draft"
    with factory() as session:
        assert source_dataset(session) == []
    assert client.get(f"/v1/passages/{pid}/review", headers=bearer("viewer-secret")).status_code == 403
    overview = client.get("/v1/curation/summary", headers=bearer("editor-secret")).json()
    assert overview["training_eligible_passages"] == 0
    assert overview["passages_by_state"]["draft"] == 1
    assert client.get("/v1/curation/summary", headers=bearer("viewer-secret")).status_code == 403


def test_certified_no_entity_passage_is_explicit_negative(env):
    client, factory, tmp = env
    uploaded = add_pdf(client, text="Nenhuma entidade científica anotada")
    pid = client.get(f"/v1/documents/{uploaded.json()['id']}/passages", headers=bearer("editor-secret")).json()[0]["id"]
    assert client.post(f"/v1/passages/{pid}/submit", headers=bearer("editor-secret")).status_code == 200
    assert client.post(f"/v1/passages/{pid}/certify", headers=bearer("reviewer-secret")).status_code == 200
    with factory() as session:
        rows = source_dataset(session)
        assert len(rows) == 1 and rows[0]["annotations"] == []
        metrics = train(session, tmp / "negative-only", epochs=1, config_override={"dim":16,"heads":4,"layers":1,"max_length":64})
        assert metrics["train_examples"] >= 1


def test_bio_segments_preserve_entire_entity_without_dropping_window():
    codec = CharTokenizer.fit(["0123456789abcdef"])
    rows = [{"sha256":"abc", "text":"0123456789abcdef", "annotations":[{"start":6,"end":11,"label":"MATERIAL"}]}]
    samples = build_samples(rows, codec, max_length=8)
    assert "".join(codec.decode(ids) for _,ids,_ in samples) == rows[0]["text"]
    entity_labels = [label for _,_,labels in samples for label in labels]
    assert entity_labels[6] == LABELS.index("B-MATERIAL")
    assert entity_labels[10] == LABELS.index("I-MATERIAL")
    assert all(labels[0] not in [LABELS.index("I-" + item) for item in ("MATERIAL", "PROCESS")] for _,_,labels in samples)
    with pytest.raises(ValueError, match="maior que max_length"):
        build_samples(rows, codec, max_length=4)
    assert _spans([LABELS.index("B-MATERIAL"), LABELS.index("I-MATERIAL"), 0]) == {(0,2,"MATERIAL")}


def test_holdout_vocabulary_does_not_leak(env):
    client, factory, tmp = env
    # 5 docs garantem ao menos um documento de treino, validação e teste.
    for i, word in enumerate(["Celulose", "Lignina", "Xilana", "Biomassa", "Amido"]):
        uploaded = add_pdf(client, text=f"{word} resultado {i}")
        pid = client.get(f"/v1/documents/{uploaded.json()['id']}/passages", headers=bearer("editor-secret")).json()[0]["id"]
        assert client.post(f"/v1/passages/{pid}/submit", headers=bearer("editor-secret")).status_code == 200
        assert client.post(f"/v1/passages/{pid}/certify", headers=bearer("reviewer-secret")).status_code == 200
    with factory() as session:
        rows = source_dataset(session)
        train(session, tmp / "holdout", epochs=1, config_override={"dim":16,"heads":4,"layers":1,"max_length":32})
        snapshot = json.loads((tmp / "holdout/dataset_snapshot.json").read_text())
        assignments = {row["document_sha256"]: row["split"] for row in snapshot}
        assert {"train","val","test"} == set(assignments.values())
        vocab = CharTokenizer.load(tmp / "holdout/tokenizer.json")
        train_chars = set("".join(row["text"] for row in rows if assignments[row["sha256"]] == "train"))
        assert set(vocab.vocab) - {"<pad>","<unk>"} == train_chars
