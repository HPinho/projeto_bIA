"""Real PostgreSQL transactional smoke tests (CI service, no mocks).

Running locally without BIA_TEST_POSTGRES_URL skips these; GitHub Actions' pg
job supplies a fresh PostgreSQL service and runs them, including two OS processes.
"""
from __future__ import annotations

from io import BytesIO
import multiprocessing
import os
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import select, text
from sqlalchemy.exc import DatabaseError

from apps.api.main import make_app
from bia_core.db import AuditEvent, Document, DocumentGovernance, PassageReview, make_session_factory
from bia_core.schema import upgrade_schema

pytestmark = pytest.mark.skipif(not os.getenv('BIA_TEST_POSTGRES_URL'), reason='PostgreSQL service required')

KEYS = {'alice': {'actor': 'alice', 'role': 'curator'},
        'bruno': {'actor': 'bruno', 'role': 'curator'},
        'admin': {'actor': 'admin', 'role': 'admin'}}


def _auth(actor):
    return {'Authorization': f'Bearer {actor}'}


def _pdf(label):
    buffer = BytesIO()
    doc = canvas.Canvas(buffer, invariant=1)
    doc.drawString(40, 700, f'Material cientifico {label}')
    doc.save()
    return buffer.getvalue()


def _worker(url, passage_id, root, barrier, queue):
    try:
        client = TestClient(make_app(db_url=url, storage_dir=root, keys=KEYS))
        barrier.wait(timeout=30)
        response = client.post(f'/v1/passages/{passage_id}/submit', headers=_auth('alice'))
        queue.put((response.status_code, response.text[:200]))
    except Exception as exc:
        queue.put((-1, repr(exc)))


def test_postgres_cas_across_processes_and_retention(tmp_path):
    url = os.environ['BIA_TEST_POSTGRES_URL']
    assert url.startswith('postgresql+')
    factory = make_session_factory(url)
    assert upgrade_schema(factory.kw['bind'], auto=False) == '0003'
    with factory() as session:
        assert session.execute(text('SELECT version FROM bia_schema_versions ORDER BY version')).scalars().all() == ['0001','0002','0003']
    client = TestClient(make_app(db_url=url, storage_dir=tmp_path / 'assets', keys=KEYS))
    uploaded = client.post('/v1/documents', headers=_auth('alice'),
        data={'title':'Estudo sob concorrencia', 'rights_basis':'open_access','training_allowed':'true'},
        files={'file': ('material.pdf', _pdf(str(uuid4())), 'application/pdf')})
    assert uploaded.status_code == 201, uploaded.text
    did = uploaded.json()['id']
    pid = client.get(f'/v1/documents/{did}/passages', headers=_auth('alice')).json()[0]['id']
    ctx = multiprocessing.get_context('spawn')
    gate, queue = ctx.Barrier(2), ctx.Queue()
    processes = [ctx.Process(target=_worker, args=(url, pid, str(tmp_path/'assets'), gate, queue)) for _ in range(2)]
    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join(timeout=45)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        assert proc.exitcode == 0, f'process crashed {proc.exitcode}'
    responses = [queue.get(timeout=5) for _ in range(2)]
    assert sorted(code for code, _ in responses) == [200, 409], responses
    with factory() as session:
        assert session.get(PassageReview, pid).status == 'submitted'
        assert len(session.scalars(select(AuditEvent).where(
            AuditEvent.action == 'passage.review.submit', AuditEvent.resource == pid)).all()) == 1
    assert client.post(f'/v1/passages/{pid}/certify', headers=_auth('bruno')).status_code == 200
    result = client.put(f'/v1/documents/{did}/governance', headers=_auth('admin'), json={
        'access_status': 'restricted', 'legal_hold': True,
        'reason': 'Restricao de custodia institucional com direito revogado'})
    assert result.status_code == 200, result.text
    with factory() as session:
        doc = session.get(Document, did)
        assert not doc.training_allowed and doc.status == 'restricted'
        assert session.get(DocumentGovernance, did).legal_hold
        assert session.get(PassageReview, pid).status == 'certified'
    assert client.get(f'/v1/documents/{did}/passages', headers=_auth('alice')).status_code == 404


def test_postgres_snapshot_trigger_blocks_mutation(tmp_path):
    url = os.environ['BIA_TEST_POSTGRES_URL']
    factory = make_session_factory(url)
    with factory() as session:
        from bia_core.db import DatasetSnapshot
        digest = uuid4().hex * 2
        row = DatasetSnapshot(sha256=digest, payload_json='{}', created_by='admin')
        session.add(row)
        session.commit()
        with pytest.raises(DatabaseError):
            session.execute(text('UPDATE dataset_snapshots SET payload_json = :p WHERE sha256 = :h'),
                            {'p':'changed', 'h': digest})
        session.rollback()
        assert session.get(DatasetSnapshot, digest).payload_json == '{}'
