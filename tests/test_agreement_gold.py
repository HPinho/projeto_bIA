"""BIA-02.5: observações cegas, concordância e gold com proveniência."""
from __future__ import annotations

from io import BytesIO
import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import select

from apps.api.main import make_app
from bia_core.agreement import _metrics, agreement_report, gold_snapshot, validate_gold_snapshot, passage_agreement_ready
from bia_core.db import AgreementResolution, AgreementVote, AuditEvent, make_session_factory
from bia_core.training import source_dataset

KEYS = {name: {"actor": name, "role": "admin" if name == "admin" else "curator" if name != "viewer" else "viewer"}
        for name in ("alice", "bruno", "carla", "diana", "admin", "viewer")}


def h(actor):
    return {"Authorization": f"Bearer {actor}"}


def pdf(text):
    buff = BytesIO()
    sheet = canvas.Canvas(buff, invariant=1)
    sheet.drawString(60, 700, text)
    sheet.save()
    return buff.getvalue()


@pytest.fixture()
def env(tmp_path):
    url = f"sqlite:///{tmp_path / 'db.sqlite3'}"
    app = make_app(db_url=url, storage_dir=tmp_path / "docs", artifact_root=tmp_path / "models", keys=KEYS)
    return TestClient(app), make_session_factory(url)


def new_passage(client, *, name="Celulose lignina teste 30 MPa", train="true"):
    r = client.post("/v1/documents", headers=h("alice"), files={"file": ("x.pdf", pdf(name), "application/pdf")},
                    data={"title": name, "rights_basis": "open_access", "training_allowed": train})
    assert r.status_code == 201, r.text
    doc = r.json()["id"]
    p = client.get(f"/v1/documents/{doc}/passages", headers=h("alice"))
    assert p.status_code == 200
    return p.json()[0]["id"]


def task(client, pid, a="alice", b="bruno", start=0, end=8):
    return client.post("/v1/agreement/tasks", headers=h("admin"),
                       json={"passage_id": pid, "start": start, "end": end, "rater_a": a, "rater_b": b})


def vote(client, task_id, who, label):
    return client.post(f"/v1/agreement/tasks/{task_id}/vote", headers=h(who), json={"label": label})


def certify(client, pid):
    assert client.post(f"/v1/passages/{pid}/submit", headers=h("alice")).status_code == 200
    assert client.post(f"/v1/passages/{pid}/certify", headers=h("bruno")).status_code == 200


def test_blind_voting_third_identity_and_no_overwrite(env):
    client, factory = env
    pid = new_passage(client)
    created = task(client, pid)
    assert created.status_code == 201, created.text
    tid = created.json()["id"]
    assert task(client, pid).status_code == 409
    assert vote(client, tid, "alice", "MATERIAL").status_code == 200
    assert vote(client, tid, "alice", "PROPERTY").status_code == 409
    assert vote(client, tid, "carla", "PROPERTY").status_code == 403
    # Nem admin nem outro avaliador pode ler a primeira opção antecipadamente.
    for actor in ("alice", "bruno", "admin"):
        info = client.get(f"/v1/agreement/tasks/{tid}", headers=h(actor))
        assert info.status_code == 200
        assert "votes" not in info.json()
    assert client.get(f"/v1/agreement/tasks/{tid}", headers=h("diana")).status_code == 403
    assert vote(client, tid, "bruno", "PROPERTY").status_code == 200
    state = client.get(f"/v1/agreement/tasks/{tid}", headers=h("admin")).json()
    assert state["needs_adjudication"] and state["votes"] == {"a": "MATERIAL", "b": "PROPERTY"}
    assert client.post(f"/v1/agreement/tasks/{tid}/adjudicate", headers=h("alice"),
                       json={"label": "NONE", "reason": "Não há suporte"}).status_code == 403
    good = client.post(f"/v1/agreement/tasks/{tid}/adjudicate", headers=h("carla"),
                       json={"label": "MATERIAL", "reason": "Conferi o artigo original"})
    assert good.status_code == 200, good.text
    assert good.json()["final_label"] == "MATERIAL"
    assert client.post(f"/v1/agreement/tasks/{tid}/adjudicate", headers=h("diana"),
                       json={"label": "NONE", "reason": "Outra decisão injustificada"}).status_code == 409
    with factory() as session:
        assert session.get(AgreementResolution, tid).actor == "carla"
        assert len(session.scalars(select(AgreementVote)).all()) == 2
        assert not passage_agreement_ready(session, pid)  # decisões divergentes ainda exigem alinhamento com gold
        assert any(x.action == "agreement.adjudicate" for x in session.scalars(select(AuditEvent)).all())


def test_pending_review_blocks_gold_and_training(env):
    client, factory = env
    pid = new_passage(client)
    certify(client, pid)
    created = task(client, pid)
    tid = created.json()["id"]
    assert client.get("/v1/datasets/gold/export", headers=h("admin")).status_code == 409
    with factory() as session:
        with pytest.raises(ValueError, match="concordância"):
            source_dataset(session)
    assert vote(client, tid, "alice", "MATERIAL").status_code == 200
    assert client.get("/v1/datasets/gold/export", headers=h("admin")).status_code == 409
    assert vote(client, tid, "bruno", "PROPERTY").status_code == 200
    assert client.get("/v1/datasets/gold/export", headers=h("admin")).status_code == 409
    assert client.post(f"/v1/agreement/tasks/{tid}/adjudicate", headers=h("carla"),
                       json={"label": "NONE", "reason": "Intervalo não corresponde à entidade"}).status_code == 200
    output = client.get("/v1/datasets/gold/export", headers=h("admin"))
    assert output.status_code == 200, output.text
    assert validate_gold_snapshot(output.json())
    assert output.json()["rows"][0]["annotations"] == []
    assert "submitted_by" not in output.json()["rows"][0]
    assert output.json()["rows"][0]["split"] == "train"
    assert client.get("/v1/datasets/gold/export", headers=h("viewer")).status_code == 403
    assert client.get("/v1/datasets/gold/export").status_code == 401
    assert client.get("/v1/datasets/gold/export", headers=h("admin")).json()["sha256"] == output.json()["sha256"]
    assert client.post(f"/v1/passages/{pid}/reopen", headers=h("carla")).status_code == 200
    assert client.get("/v1/datasets/gold/export", headers=h("admin")).json()["passages"] == 0


def test_consensus_none_and_no_auto_annotation(env):
    client, factory = env
    pid = new_passage(client)
    tid = task(client, pid).json()["id"]
    assert vote(client, tid, "alice", "NONE").status_code == 200
    assert vote(client, tid, "bruno", "NONE").status_code == 200
    state = client.get(f"/v1/agreement/tasks/{tid}", headers=h("admin")).json()
    assert state["final_label"] == "NONE" and not state["needs_adjudication"]
    assert client.post(f"/v1/agreement/tasks/{tid}/adjudicate", headers=h("carla"),
                       json={"label": "NONE", "reason": "Sem motivo para alterar"}).status_code == 409
    certify(client, pid)
    with factory() as session:
        assert source_dataset(session)[0]["annotations"] == []


def test_fixed_pair_kappa_and_non_pooled_ratios(env):
    client, factory = env
    pid = new_passage(client, name="Celulose lignina teste 30 MPa ABC DEF")
    task_ids = [task(client, pid, start=i, end=i+1).json()["id"] for i in range(4)]
    for task_id, (a, b) in zip(task_ids, [("MATERIAL", "MATERIAL"), ("PROPERTY", "PROPERTY"),
                                        ("MATERIAL", "PROPERTY"), ("PROPERTY", "MATERIAL")]):
        assert vote(client, task_id, "alice", a).status_code == 200
        assert vote(client, task_id, "bruno", b).status_code == 200
    other = task(client, pid, a="carla", b="diana", start=4, end=5).json()["id"]
    assert vote(client, other, "carla", "NONE").status_code == 200
    report = client.get("/v1/agreement/metrics", headers=h("admin"))
    assert report.status_code == 200
    data = report.json()
    assert data["tasks"] == 5 and data["pending_adjudications"] == 2
    assert len(data["by_fixed_rater_pair"]) == 2
    first = data["by_fixed_rater_pair"][0]
    assert first["tasks"] == first["complete"] == 4
    assert first["agreement_rate"] == 0.5
    assert first["expected_agreement"] == 0.5
    assert first["cohen_kappa"] == 0.0
    assert data["by_fixed_rater_pair"][1]["cohen_kappa"] is None
    assert client.get("/v1/agreement/metrics", headers=h("alice")).status_code == 403


def test_undefined_kappa_not_zero():
    report = _metrics([("NONE", "NONE"), ("NONE", "NONE")], 2)
    assert report["agreement_rate"] == 1 and report["cohen_kappa"] is None
    assert not report["kappa_defined"]
    empty = _metrics([], 3)
    assert empty["agreement_rate"] is None and empty["incomplete"] == 3


def test_bad_creation_and_rights(env):
    client, _ = env
    pid = new_passage(client)
    template = {"passage_id": pid, "start": 0, "end": 8, "rater_a": "alice", "rater_b": "bruno"}
    assert client.post("/v1/agreement/tasks", headers=h("viewer"), json=template).status_code == 403
    assert client.post("/v1/agreement/tasks", headers=h("admin"), json={**template, "end": 9999}).status_code == 422
    assert client.post("/v1/agreement/tasks", headers=h("admin"), json={**template, "rater_b": "alice"}).status_code == 422
    assert client.post("/v1/agreement/tasks", headers=h("admin"), json={**template, "rater_b": "unknown"}).status_code == 422
    assert client.get("/v1/agreement/tasks/not-found", headers=h("admin")).status_code == 404


def test_export_restricts_uncertified_and_unlicensed(env):
    client, factory = env
    licensed_pid = new_passage(client, name="Celulose MPa 49", train="true")
    no_train_pid = new_passage(client, name="Lignina MPa 51", train="false")
    draft_pid = new_passage(client, name="Polimeros rigidez 40", train="true")
    certify(client, licensed_pid)
    certify(client, no_train_pid)
    snapshot = client.get("/v1/datasets/gold/export", headers=h("admin"))
    assert snapshot.status_code == 200
    rows = snapshot.json()["rows"]
    assert len(rows) == 1 and rows[0]["passage_id"] == licensed_pid
    assert draft_pid != rows[0]["passage_id"]
    assert snapshot.json()["redistribution_allowed"] is False
    altered = snapshot.json()
    altered["rows"][0]["text"] = "dados alterados"
    assert not validate_gold_snapshot(altered)
    assert not validate_gold_snapshot({"schema": "bad", "sha256": "x"})


def test_agreement_label_must_match_certified_annotation(env):
    client, factory = env
    pid = new_passage(client, name="Celulose resistente 55 MPa")
    tid = task(client, pid).json()["id"]
    assert vote(client, tid, "alice", "MATERIAL").status_code == 200
    assert vote(client, tid, "bruno", "MATERIAL").status_code == 200
    certify(client, pid)
    assert client.get("/v1/datasets/gold/export", headers=h("admin")).status_code == 409
    assert client.post(f"/v1/passages/{pid}/reopen", headers=h("carla")).status_code == 200
    result = client.post("/v1/annotations", headers=h("alice"), json={
        "passage_id": pid, "label": "MATERIAL", "start": 0, "end": 8})
    assert result.status_code == 201
    ann_id = result.json()["id"]
    assert client.post(f"/v1/annotations/{ann_id}/review", headers=h("bruno"),
                       json={"decision": "approved"}).status_code == 200
    certify(client, pid)
    exported = client.get("/v1/datasets/gold/export", headers=h("admin"))
    assert exported.status_code == 200
    assert exported.json()["rows"][0]["annotations"] == [{"start": 0, "end": 8, "label": "MATERIAL"}]


def test_task_listing_hides_incomplete_vote(env):
    client, _ = env
    pid = new_passage(client)
    tid = task(client, pid).json()["id"]
    assert vote(client, tid, "alice", "MATERIAL").status_code == 200
    for actor in ("alice", "bruno", "admin"):
        result = client.get(f"/v1/agreement/tasks?passage_id={pid}", headers=h(actor))
        assert result.status_code == 200
        assert "votes" not in result.json()["tasks"][0]
    assert client.get(f"/v1/agreement/tasks?passage_id={pid}", headers=h("carla")).json()["tasks"] == []
    assert vote(client, tid, "bruno", "PROPERTY").status_code == 200
    assert "votes" in client.get(f"/v1/agreement/tasks?passage_id={pid}", headers=h("alice")).json()["tasks"][0]
