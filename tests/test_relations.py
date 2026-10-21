"""BIA-02.1: somente evidência humana auditada; dados sintéticos, sem afirmações científicas."""
from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from apps.api.main import make_app
from bia_core.db import Annotation, Passage, ScientificRelation, make_session_factory
from bia_core.relations import RELATION_SIGNATURES, candidate_pairs

KEYS = {
    "author": {"actor": "autor", "role": "curator", "institutes": ["inovaligno"]},
    "peer": {"actor": "revisor", "role": "curator", "institutes": ["inovaligno"]},
    "other": {"actor": "avaliador", "role": "admin"},
    "viewer": {"actor": "leitor", "role": "viewer"},
}


def auth(name):
    return {"Authorization": "Bearer " + name}


def pdf(text):
    b = BytesIO()
    c = canvas.Canvas(b, invariant=1)
    c.drawString(30, 710, text)
    c.save()
    return b.getvalue()


@pytest.fixture
def fx(tmp_path):
    url = "sqlite:///" + str(tmp_path / "db.sqlite")
    app = make_app(db_url=url, keys=KEYS, storage_dir=tmp_path / "docs", artifact_root=tmp_path / "weights")
    return TestClient(app), make_session_factory(url)


def document(client, text="Celulose submetida a prensagem apresentou rigidez de 30 MPa", rights="open_access"):
    res = client.post("/v1/documents", headers=auth("author"),
        files={"file": ("example.pdf", pdf(text), "application/pdf")},
        data={"title": "Dado sintético", "doi": "10.1234/example", "rights_basis": rights, "training_allowed": "false",
              **({"institute": "inovaligno"} if rights == "institutional_permission" else {})})
    assert res.status_code == 201, res.text
    p = client.get(f"/v1/documents/{res.json()['id']}/passages", headers=auth("author")).json()[0]
    return p


def annotation(client, passage, text, label, creator="author"):
    start = passage["text"].index(text)
    r = client.post("/v1/annotations", headers=auth(creator), json={
        "passage_id": passage["id"], "start": start, "end": start + len(text), "label": label})
    assert r.status_code == 201, r.text
    aid = r.json()["id"]
    peer = "peer" if creator == "author" else "author"
    response = client.post(f"/v1/annotations/{aid}/review", headers=auth(peer), json={"decision": "approved"})
    assert response.status_code == 200, response.text
    return aid


def certify(client, pid, submitter="author", reviewer="peer"):
    assert client.post(f"/v1/passages/{pid}/submit", headers=auth(submitter)).status_code == 200
    assert client.post(f"/v1/passages/{pid}/certify", headers=auth(reviewer)).status_code == 200


def propose(client, p, subj, obj, typ="MATERIAL_HAS_PROPERTY", who="author", begin=0, end=None):
    return client.post("/v1/relations", headers=auth(who), json={
        "passage_id": p["id"], "subject_annotation_id": subj, "object_annotation_id": obj,
        "relation_type": typ, "evidence_start": begin,
        "evidence_end": len(p["text"]) if end is None else end})


def test_candidates_need_certification_and_approved_anchors(fx):
    c, sf = fx
    p = document(c)
    material = annotation(c, p, "Celulose", "MATERIAL")
    prop = annotation(c, p, "rigidez", "PROPERTY")
    with sf() as session:
        assert candidate_pairs(session, p["id"]) == []
    assert c.get(f"/v1/passages/{p['id']}/relation-candidates", headers=auth("author")).json()["candidates"] == []
    assert propose(c, p, material, prop).status_code == 409
    certify(c, p["id"])
    result = c.get(f"/v1/passages/{p['id']}/relation-candidates", headers=auth("author")).json()
    assert result["status"] == "candidate_unverified"
    found = [i for i in result["candidates"] if i["relation_type"] == "MATERIAL_HAS_PROPERTY"]
    assert len(found) == 1 and found[0]["status"] == "candidate_unverified"
    assert "Celulose" in found[0]["evidence_quote"] and "rigidez" in found[0]["evidence_quote"]
    assert c.get("/v1/relations/search", headers=auth("viewer")).json()["results"] == []


def test_relation_independent_review_search_and_audit(fx):
    c, sf = fx
    p = document(c)
    s = annotation(c, p, "Celulose", "MATERIAL")
    t = annotation(c, p, "rigidez", "PROPERTY")
    certify(c, p["id"])
    proposal = propose(c, p, s, t)
    assert proposal.status_code == 201, proposal.text
    rid = proposal.json()["id"]
    assert c.post(f"/v1/relations/{rid}/review", headers=auth("author"), json={"decision": "approved"}).status_code == 403
    assert c.get("/v1/relations/search?q=celulose", headers=auth("viewer")).json()["results"] == []
    verified = c.post(f"/v1/relations/{rid}/review", headers=auth("peer"), json={"decision": "approved"})
    assert verified.status_code == 200 and verified.json()["verified"] is True
    result = c.get("/v1/relations/search?q=rigidez", headers=auth("viewer")).json()
    assert result["status"] == "ok" and result["scientifically_validated"] is False
    assert len(result["results"]) == 1
    rel = result["results"][0]
    assert rel["id"] == rid and rel["doi"] == "10.1234/example" and rel["page"] == 1
    assert rel["evidence"]["quote"] == p["text"][rel["evidence"]["start"]:rel["evidence"]["end"]]
    assert rel["subject"]["quote"] == "Celulose" and rel["object"]["quote"] == "rigidez"
    assert rel["review"] == {"proposed_by": "autor", "reviewed_by": "revisor"}
    assert c.get("/v1/relations/search?q=ausente", headers=auth("viewer")).json()["status"] == "insufficient_evidence"
    assert c.get("/v1/relations/search?relation_type=BAD", headers=auth("viewer")).status_code == 422
    with sf() as session:
        assert session.get(ScientificRelation, rid).status == "approved"
    assert propose(c, p, s, t).status_code == 409
    assert c.post(f"/v1/relations/{rid}/review", headers=auth("other"), json={"decision": "approved"}).status_code == 409


def test_relations_fail_closed_on_reopen_and_recertification(fx):
    c, _ = fx
    p = document(c)
    s = annotation(c, p, "Celulose", "MATERIAL")
    t = annotation(c, p, "rigidez", "PROPERTY")
    certify(c, p["id"])
    rid = propose(c, p, s, t).json()["id"]
    assert c.post(f"/v1/relations/{rid}/review", headers=auth("peer"), json={"decision": "approved"}).json()["verified"]
    c.post(f"/v1/passages/{p['id']}/reopen", headers=auth("author"))
    assert c.get("/v1/relations/search", headers=auth("viewer")).json()["results"] == []
    certify(c, p["id"])
    assert c.get("/v1/relations/search", headers=auth("viewer")).json()["results"] == []
    rows = c.get(f"/v1/passages/{p['id']}/relations", headers=auth("author")).json()["relations"]
    assert len(rows) == 1 and rows[0]["status"] == "approved" and rows[0]["certification_current"] is False
    new_rel = propose(c, p, s, t)
    assert new_rel.status_code == 201 and new_rel.json()["id"] != rid


def test_relation_rejects_wrong_direction_and_ungrounded_evidence(fx):
    c, _ = fx
    p = document(c)
    m = annotation(c, p, "Celulose", "MATERIAL")
    prop = annotation(c, p, "rigidez", "PROPERTY")
    certify(c, p["id"])
    assert propose(c, p, prop, m).status_code == 422
    assert propose(c, p, m, m).status_code == 422
    assert propose(c, p, m, prop, end=8).status_code == 422
    assert propose(c, p, m, prop, end=99999).status_code == 422
    assert propose(c, p, m, "missing").status_code == 422
    assert len(RELATION_SIGNATURES) >= 7


def test_rejected_relation_never_appears_in_public_graph(fx):
    c, _ = fx
    p = document(c)
    m = annotation(c, p, "Celulose", "MATERIAL")
    prop = annotation(c, p, "rigidez", "PROPERTY")
    certify(c, p["id"])
    rid = propose(c, p, m, prop).json()["id"]
    result = c.post(f"/v1/relations/{rid}/review", headers=auth("peer"), json={"decision": "rejected"})
    assert result.status_code == 200 and result.json()["verified"] is False
    assert c.get("/v1/relations/search", headers=auth("viewer")).json()["status"] == "insufficient_evidence"


def test_restricted_document_relations_respect_visibility(fx):
    c, _ = fx
    p = document(c, rights="institutional_permission")
    m = annotation(c, p, "Celulose", "MATERIAL")
    prop = annotation(c, p, "rigidez", "PROPERTY")
    certify(c, p["id"])
    rid = propose(c, p, m, prop).json()["id"]
    c.post(f"/v1/relations/{rid}/review", headers=auth("peer"), json={"decision": "approved"})
    assert c.get("/v1/relations/search", headers=auth("viewer")).json()["results"] == []
    assert len(c.get("/v1/relations/search", headers=auth("other")).json()["results"]) == 1
    assert c.get(f"/v1/passages/{p['id']}/relation-candidates", headers=auth("viewer")).status_code == 403
    assert c.get("/v1/relations/search").status_code == 401
    assert propose(c, p, m, prop, who="viewer").status_code == 403


def test_stale_pending_relation_cannot_be_approved(fx):
    c, _ = fx
    p = document(c)
    m = annotation(c, p, "Celulose", "MATERIAL")
    prop = annotation(c, p, "rigidez", "PROPERTY")
    certify(c, p["id"])
    rid = propose(c, p, m, prop).json()["id"]
    c.post(f"/v1/passages/{p['id']}/reopen", headers=auth("author"))
    certify(c, p["id"])
    response = c.post(f"/v1/relations/{rid}/review", headers=auth("peer"), json={"decision": "approved"})
    assert response.status_code == 409


def test_cross_passage_anchors_are_rejected(fx):
    c, sf = fx
    p = document(c)
    q = document(c, text="Polimero possui dureza de 17 MPa")
    m = annotation(c, p, "Celulose", "MATERIAL")
    other = annotation(c, q, "dureza", "PROPERTY")
    certify(c, p["id"])
    certify(c, q["id"])
    assert propose(c, p, m, other).status_code == 422
    with sf() as session:
        assert session.query(ScientificRelation).count() == 0
