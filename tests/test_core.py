from __future__ import annotations

from io import BytesIO

import pytest
from reportlab.pdfgen import canvas
from sqlalchemy import select
from fastapi.testclient import TestClient

from apps.api.main import make_app
from bia_core.db import Annotation, Document, Passage, make_session_factory
from bia_core.retrieval import bm25, tokenize
from bia_core.tokenizer import CharTokenizer
from bia_core.training import document_splits, extract, train


KEYS = {
    "editor-secret": {"actor": "editor", "role": "curator"},
    "reviewer-secret": {"actor": "reviewer", "role": "curator"},
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
                       data={"title": "Artigo de demonstração", "rights_basis": rights, "training_allowed": training})


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
        output = tmp_path / "run"
        metrics = train(session, output, epochs=2, seed=42, config_override={"dim": 16, "heads": 4, "layers": 1, "max_length": 64})
        assert metrics["train_examples"] > 0
        assert metrics["validation_micro_f1"] is None
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
    with factory() as session:
        with pytest.raises(ValueError, match="Não há anotações"):
            train(session, tmp_path / "should-not-exist")
    assert client.get(f"/v1/extract/{pid}", headers=bearer("editor-secret")).json()["status"] == "insufficient_evidence"


def test_splits_are_by_document_not_passage():
    items = [{"sha256": hex(i), "text": "x"} for i in range(12)]
    items += [{"sha256": "0x1", "text": "y"}]
    splits = document_splits(items)
    assert len(splits) == 12
    assert {"train", "val", "test"} == set(splits.values())
    assert splits["0x1"] == splits[items[1]["sha256"]]
