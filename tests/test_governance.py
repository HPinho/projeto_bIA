"""BIA-02.7: restricted sources, retention controls and CAS curation."""
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import select, text

from apps.api.main import make_app
from bia_core.db import (AuditEvent, Document, DocumentGovernance, Passage, PassageReview,
                         make_session_factory)
from bia_core.schema import upgrade_schema
from bia_core.training import source_dataset
from bia_core.transitions import certify_once, reopen_once, submit_once


KEYS = {
    'admin': {'actor': 'admin', 'role': 'admin'},
    'alice': {'actor': 'alice', 'role': 'curator'},
    'bruno': {'actor': 'bruno', 'role': 'curator'},
    'reader': {'actor': 'reader', 'role': 'viewer'},
}


def auth(actor):
    return {'Authorization': f'Bearer {actor}'}


def pdf(text_value='Celulose pode formar compositos 45 MPa'):
    buffer = BytesIO()
    page = canvas.Canvas(buffer, invariant=1)
    page.drawString(20, 700, text_value)
    page.save()
    return buffer.getvalue()


@pytest.fixture()
def environment(tmp_path):
    url = f'sqlite:///{tmp_path / "database.sqlite"}'
    app = make_app(db_url=url, storage_dir=tmp_path / 'docs', artifact_root=tmp_path / 'weights', keys=KEYS)
    return TestClient(app), make_session_factory(url)


def upload(client):
    response = client.post('/v1/documents', headers=auth('alice'),
        data={'title': 'Materiais de pesquisa', 'rights_basis': 'open_access', 'training_allowed': 'true'},
        files={'file': ('material.pdf', pdf(), 'application/pdf')})
    assert response.status_code == 201, response.text
    return response.json()['id']


def test_governance_restrict_revokes_training_and_snapshot_access(environment):
    client, factory = environment
    did = upload(client)
    pid = client.get(f'/v1/documents/{did}/passages', headers=auth('alice')).json()[0]['id']
    assert client.post(f'/v1/passages/{pid}/submit', headers=auth('alice')).status_code == 200
    assert client.post(f'/v1/passages/{pid}/certify', headers=auth('bruno')).status_code == 200
    result = client.post('/v1/datasets/gold/snapshots', headers=auth('admin'))
    assert result.status_code == 201, result.text
    digest = result.json()['sha256']
    before = client.get(f'/v1/datasets/gold/snapshots/{digest}', headers=auth('admin'))
    assert before.status_code == 200
    governance = client.put(f'/v1/documents/{did}/governance', headers=auth('admin'), json={
        'access_status': 'restricted', 'legal_hold': True,
        'retention_until': (datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
        'reason': 'Solicitacao formal de interrupcao do uso para treinamento'})
    assert governance.status_code == 200, governance.text
    assert governance.json()['training_allowed'] is False
    assert governance.json()['physical_deletion_supported'] is False
    assert client.get(f'/v1/documents/{did}', headers=auth('reader')).status_code == 404
    assert client.get(f'/v1/documents/{did}/passages', headers=auth('alice')).status_code == 404
    assert client.get(f'/v1/documents/{did}', headers=auth('admin')).status_code == 200
    assert client.get('/v1/search?q=celulose', headers=auth('reader')).json()['results'] == []
    assert client.get(f'/v1/datasets/gold/snapshots/{digest}', headers=auth('admin')).status_code == 403
    with factory() as session:
        assert source_dataset(session) == []
        assert session.get(DocumentGovernance, did).legal_hold is True
        assert len(session.scalars(select(AuditEvent).where(AuditEvent.action == 'document.governance.update')).all()) == 1
        assert session.get(Document, did).asset_path.endswith('.pdf')


def test_hold_release_explicit_and_no_implicit_training_regrant(environment):
    client, factory = environment
    did = upload(client)
    url = f'/v1/documents/{did}/governance'
    hold = {'access_status': 'restricted', 'legal_hold': True,
            'reason': 'Preservacao institucional solicitada pelo responsavel'}
    assert client.put(url, headers=auth('alice'), json=hold).status_code == 403
    assert client.put(url, headers=auth('admin'), json=hold).status_code == 200
    request = {**hold, 'access_status': 'active', 'legal_hold': False}
    assert client.put(url, headers=auth('admin'), json=request).status_code == 409
    assert client.put(url, headers=auth('admin'), json={**request, 'release_hold': True}).status_code == 200
    assert client.get(f'/v1/documents/{did}', headers=auth('reader')).status_code == 200
    assert client.get(url, headers=auth('alice')).status_code == 403
    with factory() as session:
        doc = session.get(Document, did)
        assert doc.status == 'ready' and not doc.training_allowed
        assert session.get(DocumentGovernance, did).legal_hold is False


def test_retention_report_counts_due_and_rejects_naive_date(environment):
    client, factory = environment
    did = upload(client)
    url = f'/v1/documents/{did}/governance'
    assert client.put(url, headers=auth('admin'), json={
        'access_status': 'active', 'reason': 'Revisao documental para protocolo de custodia',
        'retention_until': '2030-01-01T00:00:00'}).status_code == 422
    response = client.put(url, headers=auth('admin'), json={
        'access_status': 'active', 'reason': 'Revisao documental para protocolo de custodia',
        'retention_until': (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()})
    assert response.status_code == 200
    result = client.get('/v1/corpus/readiness', headers=auth('admin'))
    assert result.status_code == 200
    assert result.json()['documents'] == 1
    assert result.json()['retention_review_due'] == 1
    assert result.json()['training_authorized_ready_documents'] == 1
    assert client.get('/v1/corpus/readiness', headers=auth('reader')).status_code == 403
    assert client.get('/v1/corpus/readiness').status_code == 401


def test_cas_submit_certify_reopen_and_stale_requests(environment):
    client, factory = environment
    did = upload(client)
    pid = client.get(f'/v1/documents/{did}/passages', headers=auth('alice')).json()[0]['id']
    with factory() as session:
        assert submit_once(session, pid, 'alice')
        session.commit()
    with factory() as session:
        assert not submit_once(session, pid, 'bruno')
        assert not certify_once(session, pid, 'alice')
        assert certify_once(session, pid, 'bruno')
        session.commit()
    with factory() as session:
        assert not certify_once(session, pid, 'bruno')
        assert reopen_once(session, pid)
        assert not reopen_once(session, pid)
        assert submit_once(session, pid, 'alice')
        session.commit()
    with factory() as session:
        row = session.get(PassageReview, pid)
        assert row.status == 'submitted' and row.certified_by is None
        assert row.submitted_by == 'alice'
    assert client.post(f'/v1/passages/{pid}/certify', headers=auth('bruno')).status_code == 200
    assert client.post(f'/v1/passages/{pid}/certify', headers=auth('bruno')).status_code == 409
    assert client.post(f'/v1/passages/{pid}/reopen', headers=auth('bruno')).status_code == 200
    assert client.post(f'/v1/passages/{pid}/reopen', headers=auth('bruno')).status_code == 409


def test_0002_to_0003_upgrade_preserves_snapshots(environment):
    _, factory = environment
    engine = factory.kw['bind']
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM bia_schema_versions WHERE version='0003'"))
        conn.execute(text('DROP TABLE document_governance'))
    assert upgrade_schema(engine) == '0003'
    with factory() as session:
        assert session.execute(text('SELECT version FROM bia_schema_versions ORDER BY version')).scalars().all() == ['0001','0002','0003']


def test_manifest_excludes_raw_text_and_lists_training_obstacles(environment):
    client, factory = environment
    did = upload(client)
    response = client.get('/v1/corpus/manifest?limit=5', headers=auth('admin'))
    assert response.status_code == 200
    assert len(response.json()['items']) == 1
    item = response.json()['items'][0]
    assert item['id'] == did and item['certified_passages'] == 0
    assert 'no_certified_passages' in item['readiness_issues']
    assert not any(k in item for k in ('text', 'asset_path', 'annotations', 'pdf'))
    assert client.get('/v1/corpus/manifest', headers=auth('alice')).status_code == 403
    assert client.get('/v1/corpus/manifest?limit=101', headers=auth('admin')).status_code == 422
    blocked = client.put(f'/v1/documents/{did}/governance', headers=auth('admin'), json={
        'access_status':'restricted', 'legal_hold': True,
        'reason':'Instituicao solicitou suspensao preventiva de acesso'})
    assert blocked.status_code == 200
    item = client.get('/v1/corpus/manifest', headers=auth('admin')).json()['items'][0]
    assert {'access_restricted', 'training_not_authorized'}.issubset(item['readiness_issues'])
