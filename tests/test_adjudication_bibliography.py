"""BIA-02.4: revisão tripartite e desduplicação sem fundir evidências."""
from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import select

from apps.api.main import make_app
from bia_core.db import Annotation, AnnotationDispute, AuditEvent, BibliographicDecision, Document, make_session_factory
from bia_core.bibliography import normalize_doi, title_key, publication_clusters
from bia_core.training import document_splits, source_dataset
from bia_core.relation_learning import grouped_splits

KEYS = {
    "alice": {"role": "curator", "actor": "alice", "institutes": ["inovaligno"]},
    "bruno": {"role": "curator", "actor": "bruno", "institutes": ["inovaligno"]},
    "carla": {"role": "curator", "actor": "carla", "institutes": ["inovaligno"]},
    "administrator": {"role": "admin", "actor": "administrator"},
    "reader": {"role": "viewer", "actor": "reader"},
}


def h(name):
    return {"Authorization": f"Bearer {name}"}


def pdf(text):
    buffer = BytesIO()
    document = canvas.Canvas(buffer, invariant=1)
    document.drawString(60, 700, text)
    document.save()
    return buffer.getvalue()


@pytest.fixture()
def env(tmp_path):
    url = f"sqlite:///{tmp_path / 'db.sqlite3'}"
    app = make_app(db_url=url, storage_dir=tmp_path / "storage", artifact_root=tmp_path / "weights", keys=KEYS)
    return TestClient(app), make_session_factory(url)


def insert(client, title, text, doi=None, year=None, rights="open_access", actor="alice"):
    data = {"title": title, "rights_basis": rights, "training_allowed": "true"}
    if rights == "institutional_permission":
        data["institute"] = "inovaligno"
    if doi:
        data["doi"] = doi
    if year is not None:
        data["year"] = str(year)
    response = client.post("/v1/documents", headers=h(actor),
                           files={"file": ("article.pdf", pdf(text), "application/pdf")}, data=data)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def passage(client, doc):
    return client.get(f"/v1/documents/{doc}/passages", headers=h("alice")).json()[0]["id"]


def open_case(client, pid, user="alice", label="MATERIAL", start=0, end=8):
    return client.post("/v1/adjudications", headers=h(user), json={"passage_id": pid,
            "alternative": {"label": label, "start": start, "end": end}})


def second(client, case, user="bruno", label="PROPERTY", start=9, end=17):
    return client.post(f"/v1/adjudications/{case}/second", headers=h(user),
                       json={"label": label, "start": start, "end": end})


def resolve(client, case, user="carla", resolution="first", reason="Original do artigo confirma a leitura"):
    return client.post(f"/v1/adjudications/{case}/resolve", headers=h(user),
                       json={"resolution": resolution, "reason": reason})


def test_doi_normalization_not_false_match():
    assert normalize_doi(" https://doi.org/10.1000/ABC.123 ") == "10.1000/abc.123"
    assert normalize_doi("DOI:10.1000/abc.123") == "10.1000/abc.123"
    assert normalize_doi("doi:garbage") is None
    assert normalize_doi("10.1234/") is None
    assert normalize_doi(None) is None
    assert title_key("Ácido – CELULÓSICO") == "acido celulosico"


def test_dispute_three_independent_people_and_certification_block(env):
    client, factory = env
    document = insert(client, "Experiment A", "Celulose resistencia 45 MPa")
    pid = passage(client, document)
    opened = open_case(client, pid)
    assert opened.status_code == 201, opened.text
    case = opened.json()["id"]
    assert client.post(f"/v1/passages/{pid}/submit", headers=h("alice")).status_code == 409
    assert second(client, case, user="alice").status_code == 403
    assert resolve(client, case).status_code == 409
    response = second(client, case)
    assert response.status_code == 200, response.text
    assert second(client, case).status_code == 409
    assert resolve(client, case, user="alice").status_code == 403
    assert resolve(client, case, user="bruno").status_code == 403
    approved = resolve(client, case)
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "resolved"
    assert approved.json()["reason"]
    assert resolve(client, case).status_code == 409
    with factory() as session:
        ann = session.get(Annotation, approved.json()["annotation_id"])
        assert ann.actor == "alice" and ann.reviewer == "carla" and ann.status == "approved"
        assert session.get(AnnotationDispute, case).reason == "Original do artigo confirma a leitura"
    assert client.post(f"/v1/passages/{pid}/submit", headers=h("alice")).status_code == 200
    assert client.post(f"/v1/passages/{pid}/certify", headers=h("bruno")).status_code == 200
    with factory() as session:
        rows = source_dataset(session)
        assert len(rows) == 1 and len(rows[0]["annotations"]) == 1
        assert rows[0]["annotations"][0]["label"] == "MATERIAL"


def test_neither_choice_creates_no_unjustified_label(env):
    client, factory = env
    pid = passage(client, insert(client, "Experimental B", "Celulose lignina processada"))
    case = open_case(client, pid).json()["id"]
    assert second(client, case).status_code == 200
    answer = resolve(client, case, resolution="neither")
    assert answer.status_code == 200
    assert answer.json()["annotation_id"] is None
    with factory() as session:
        assert session.scalars(select(Annotation)).all() == []
    assert client.post(f"/v1/passages/{pid}/submit", headers=h("alice")).status_code == 200
    assert client.post(f"/v1/passages/{pid}/certify", headers=h("bruno")).status_code == 200
    with factory() as session:
        assert source_dataset(session)[0]["annotations"] == []


def test_conflict_does_not_overwrite_annotations_or_resolve(env):
    client, factory = env
    pid = passage(client, insert(client, "Experimental C", "Celulose lignina extraida"))
    case = open_case(client, pid).json()["id"]
    assert second(client, case).status_code == 200
    a = client.post("/v1/annotations", headers=h("bruno"), json={"passage_id": pid,
        "label": "MATERIAL", "start": 0, "end": 8})
    assert a.status_code == 201
    assert resolve(client, case).status_code == 409
    with factory() as session:
        assert session.get(AnnotationDispute, case).status == "awaiting_adjudication"
    # The adjudicator can reject both alternatives without changing existing labels.
    assert resolve(client, case, resolution="neither").status_code == 200
    assert client.post(f"/v1/passages/{pid}/submit", headers=h("alice")).status_code == 409  # existing pending annotation


def test_validation_rights_and_offsets_for_disputes(env):
    client, factory = env
    restricted = insert(client, "Protected material", "Celulose lignina extraida", rights="institutional_permission")
    pid = passage(client, restricted)
    assert open_case(client, pid, end=1000).status_code == 422
    case = open_case(client, pid).json()["id"]
    assert second(client, case, start=19, end=190).status_code == 422
    assert client.get(f"/v1/passages/{pid}/adjudications", headers=h("reader")).status_code == 403
    assert client.get(f"/v1/passages/{pid}/adjudications", headers=h("carla")).json()["unresolved"] == 1
    assert client.post(f"/v1/passages/{pid}/submit", headers=h("bruno")).status_code == 409


def test_find_duplicates_without_merging_and_independent_review(env):
    client, factory = env
    a = insert(client, "Propriedades de compósitos de celulose", "Alpha 41 MPa", doi="10.1234/abc", year=2024)
    b = insert(client, "Propriedades de compositos de celulose", "Beta  42 MPa", doi="HTTPS://DOI.ORG/10.1234/ABC", year=2024)
    c = insert(client, "Propriedades de compositos de celulose", "Gamma 43 MPa", doi="10.1234/other", year=2024)
    found = client.get(f"/v1/documents/{a}/duplicates", headers=h("bruno"))
    assert found.status_code == 200
    hits = {x["document_id"]: x for x in found.json()["candidates"]}
    assert "same_normalized_doi" in hits[b]["signals"]
    assert "same_normalized_title_and_year" in hits[c]["signals"]
    assert found.json()["automatically_merged"] is False
    assert client.post("/v1/bibliography/review", headers=h("alice"), json={
        "left_document_id": a, "right_document_id": b, "verdict": "same_publication", "reason": "DOI coincide nos dois registros"}).status_code == 403
    reviewed = client.post("/v1/bibliography/review", headers=h("bruno"), json={
        "left_document_id": b, "right_document_id": a, "verdict": "same_publication", "reason": "DOI coincide nos dois registros"})
    assert reviewed.status_code == 200, reviewed.text
    assert client.post("/v1/bibliography/review", headers=h("carla"), json={
        "left_document_id": a, "right_document_id": b, "verdict": "distinct_publication", "reason": "Tentativa de sobrescrita"}).status_code == 409
    with factory() as session:
        assert len(session.scalars(select(Document)).all()) == 3
        assert len(session.scalars(select(BibliographicDecision)).all()) == 1
        group = publication_clusters(session)
        assert group[a] == group[b]
        assert group[c] != group[a]


def test_confirmed_same_publication_groups_different_dois_for_holdouts(env):
    client, factory = env
    title = "Análise comparativa de polímeros experimentais"
    a = insert(client, title, "Celulose experimental grupo A", doi="10.1234/x1", year=2025)
    b = insert(client, title, "Celulose experimental grupo B", doi="10.1234/x2", year=2025)
    with factory() as session:
        groups = publication_clusters(session)
        assert groups[a] != groups[b]
    assert client.post("/v1/bibliography/review", headers=h("bruno"), json={
        "left_document_id": a, "right_document_id": b, "verdict": "same_publication", "reason": "Mesma publicação em versões distintas"}).status_code == 200
    with factory() as session:
        groups = publication_clusters(session)
        assert groups[a] == groups[b]
        rows = [{"sha256": str(i), "publication_group": f"solo{i}"} for i in range(12)]
        rows.extend([{"sha256": "copy1", "publication_group": groups[a]},
                     {"sha256": "copy2", "publication_group": groups[b]}])
        assign = document_splits(rows)
        assert assign["copy1"] == assign["copy2"]
        relations = [{"document_sha256": row["sha256"], "passage_sha256": "pass"+row["sha256"],
                      "publication_group": row["publication_group"], "label": i % 2}
                     for i, row in enumerate(rows)]
        rs = grouped_splits(relations, 42)
        assert rs["copy1"] == rs["copy2"]


def test_reject_unrelated_pairs_and_invalid_doi(env):
    client, _ = env
    a = insert(client, "Livro completamente diferente A", "Celulose 9 MPa", doi="10.1234/a")
    b = insert(client, "Texto totalmente distante", "Lignina 45 MPa", doi="10.1234/b")
    assert client.get(f"/v1/documents/{a}/duplicates", headers=h("carla")).json()["candidates"] == []
    assert client.post("/v1/bibliography/review", headers=h("carla"), json={
        "left_document_id": a, "right_document_id": b, "verdict": "same_publication", "reason": "Sem base suficiente para comparação"}).status_code == 422
    bad = client.post("/v1/documents", headers=h("alice"), files={"file": ("paper.pdf", pdf("outro artigo"), "application/pdf")},
                      data={"title": "Documento inválido", "rights_basis": "open_access", "doi": "não-é-doi"})
    assert bad.status_code == 422
