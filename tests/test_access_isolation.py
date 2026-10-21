"""BIA-02.8: prove there is no cross-institute visibility without membership."""
from io import BytesIO
import os

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import inspect, select, text

from apps.api.main import make_app
from bia_core.access import authenticate, compile_keys
from bia_core.db import Document, AuditEvent, make_session_factory
from bia_core.schema import upgrade_schema

KEYS = {
    'lig-admin': {'actor': 'supervisor', 'role': 'admin'},
    'lig-ligno': {'actor': 'ligno', 'role': 'curator', 'institutes': ['inovaligno']},
    'lig-peer': {'actor': 'peer', 'role': 'curator', 'institutes': ['inovaligno']},
    'lig-poly': {'actor': 'poly', 'role': 'curator', 'institutes': ['polissacarideos']},
    'lig-circ': {'actor': 'circ', 'role': 'curator', 'institutes': ['circularidade']},
    'lig-reader': {'actor': 'reader', 'role': 'viewer', 'institutes': ['inovaligno']},
    'lig-unbound': {'actor': 'unbound', 'role': 'curator'},
}


def auth(who):
    return {'Authorization': f'Bearer lig-{who}'}


def document_pdf(line):
    stream = BytesIO()
    page = canvas.Canvas(stream, invariant=1)
    page.drawString(40, 690, line)
    page.save()
    return stream.getvalue()


@pytest.fixture()
def sandbox(tmp_path):
    url = f'sqlite:///{tmp_path / "test.sqlite3"}'
    api = TestClient(make_app(db_url=url, keys=KEYS,
                              storage_dir=tmp_path / 'assets', artifact_root=tmp_path / 'models'))
    return api, make_session_factory(url)


def put(client, actor, content, *, institute=None, rights='institutional_permission'):
    data = {'title': f'Study {content}', 'training_allowed': 'false', 'rights_basis': rights}
    if institute:
        data['institute'] = institute
    return client.post('/v1/documents', headers=auth(actor), data=data,
                       files={'file': ('study.pdf', document_pdf(content), 'application/pdf')})


def test_authentication_is_scoped_and_invalid_profiles_rejected(sandbox):
    api, _ = sandbox
    anonymous = api.get('/v1/auth/me')
    assert anonymous.status_code == 401
    assert api.get('/v1/auth/me', headers=auth('unknown')).status_code == 401
    profile = api.get('/v1/auth/me', headers=auth('ligno')).json()
    assert profile == {'actor': 'ligno', 'role': 'curator', 'institutes': ['inovaligno'],
                       'auth_method': 'bearer-local'}
    assert 'lig-ligno' not in str(profile)
    assert authenticate(compile_keys(KEYS), 'lig-ligno')['actor'] == 'ligno'
    assert authenticate(compile_keys(KEYS), 'lig-wrong') is None
    with pytest.raises(ValueError, match='duplicada'):
        compile_keys({'one': {'actor': 'bob', 'role': 'curator'},
                      'two': {'actor': 'bob', 'role': 'curator'}})
    with pytest.raises(ValueError, match='Afiliação'):
        compile_keys({'one': {'actor': 'bob', 'role': 'curator', 'institutes': ['untrusted']}})
    with pytest.raises(ValueError, match='Afiliação'):
        compile_keys({'one': {'actor': 'bob', 'role': 'curator', 'institutes': [['inovaligno']]}})


def test_institutional_upload_requires_real_membership(sandbox):
    api, _ = sandbox
    assert put(api, 'unbound', 'Unbound', institute='inovaligno').status_code == 403
    assert put(api, 'ligno', 'No assigned institute').status_code == 422
    assert put(api, 'ligno', 'Outside affiliation', institute='polissacarideos').status_code == 403
    assert put(api, 'ligno', 'Unknown institution', institute='unknown').status_code == 422
    assert put(api, 'ligno', 'Open data marked institutional', rights='open_access', institute='inovaligno').status_code == 422
    assert put(api, 'ligno', 'Ligno 45 MPa', institute='inovaligno').status_code == 201


def test_documents_passages_search_and_curation_isolated(sandbox):
    api, factory = sandbox
    ligno = put(api, 'ligno', 'Alfacelulose 45 MPa', institute='inovaligno')
    poly = put(api, 'poly', 'Betaxilana 60 MPa', institute='polissacarideos')
    assert ligno.status_code == poly.status_code == 201
    a, b = ligno.json()['id'], poly.json()['id']
    assert api.get(f'/v1/documents/{a}', headers=auth('poly')).status_code == 404
    assert api.get(f'/v1/documents/{b}/passages', headers=auth('ligno')).status_code == 404
    assert api.get(f'/v1/documents/{a}/passages', headers=auth('reader')).status_code == 404
    assert api.get(f'/v1/documents/{a}/passages', headers=auth('unbound')).status_code == 404
    assert api.get(f'/v1/documents/{a}/passages', headers=auth('ligno')).status_code == 200
    assert api.get(f'/v1/documents/{b}/passages', headers=auth('poly')).status_code == 200
    assert api.get(f'/v1/documents/{b}', headers=auth('admin')).status_code == 200
    for outsider, token in [('poly', 'alfacelulose'), ('ligno', 'betaxilana'), ('circ', 'betaxilana')]:
        hits = api.get('/v1/search?q=' + token, headers=auth(outsider)).json()
        assert hits['status'] == 'insufficient_evidence' and not hits['results']
    assert api.get('/v1/search?q=alfacelulose', headers=auth('ligno')).json()['results'][0]['document_id'] == a
    assert api.get('/v1/search?q=betaxilana', headers=auth('poly')).json()['results'][0]['document_id'] == b
    scoped = api.get('/v1/curation/summary', headers=auth('ligno')).json()
    assert scoped['documents'] == scoped['passages'] == 1
    assert api.get(f'/v1/documents/{b}/duplicates', headers=auth('ligno')).status_code == 404
    # Global deduplication cannot disclose the document identifier to a foreign institute.
    copied = api.post('/v1/documents', headers=auth('ligno'),
                      data={'title': 'Copied', 'rights_basis': 'institutional_permission', 'institute': 'inovaligno'},
                      files={'file': ('a.pdf', document_pdf('Betaxilana 60 MPa'), 'application/pdf')})
    assert copied.status_code == 409 and b not in copied.text
    with factory() as db:
        assert db.get(Document, a).institute == 'inovaligno'
        assert db.get(Document, b).institute == 'polissacarideos'


def test_legacy_unclassified_fails_closed_until_admin_classifies(sandbox):
    api, factory = sandbox
    created = put(api, 'admin', 'Unassigned legacy 82 MPa', institute='circularidade')
    assert created.status_code == 201
    doc_id = created.json()['id']
    with factory.begin() as db:
        db.get(Document, doc_id).institute = None
    assert api.get(f'/v1/documents/{doc_id}', headers=auth('circ')).status_code == 404
    assert api.get('/v1/search?q=Unassigned', headers=auth('circ')).json()['results'] == []
    missing = api.put(f'/v1/documents/{doc_id}/institution', headers=auth('ligno'),
                      json={'institute': 'circularidade', 'reason': 'Legacy rights reviewed and assigned'})
    assert missing.status_code == 403
    assigned = api.put(f'/v1/documents/{doc_id}/institution', headers=auth('admin'),
                       json={'institute': 'circularidade', 'reason': 'Institutional agreement was verified'})
    assert assigned.status_code == 200
    assert api.get(f'/v1/documents/{doc_id}', headers=auth('circ')).status_code == 200
    assert api.get(f'/v1/documents/{doc_id}', headers=auth('poly')).status_code == 404
    assert api.put(f'/v1/documents/{doc_id}/institution', headers=auth('admin'),
                   json={'institute': 'polissacarideos', 'reason': 'Attempt to bypass approved classification'}).status_code == 409
    with factory() as db:
        records = list(db.scalars(select(AuditEvent).where(AuditEvent.action == 'document.institution.assign')))
        assert len(records) == 1 and doc_id in records[0].resource


def test_version_0003_migrates_column_without_losing_documents(tmp_path):
    url = f'sqlite:///{tmp_path / "legacy.sqlite3"}'
    factory = make_session_factory(url)
    with factory.begin() as db:
        db.add(Document(id='a' * 32, title='Original', rights_basis='institutional_permission',
                        training_allowed=False, sha256='b' * 64, asset_path='old.pdf', status='ready'))
    engine = factory.kw['bind']
    with engine.begin() as conn:
        conn.exec_driver_sql('DROP INDEX IF EXISTS ix_documents_institute')
        conn.exec_driver_sql('ALTER TABLE documents DROP COLUMN institute')
        conn.execute(text("DELETE FROM bia_schema_versions WHERE version='0004'"))
    assert 'institute' not in {col['name'] for col in inspect(engine).get_columns('documents')}
    with pytest.raises(RuntimeError, match='Schema desatualizado'):
        upgrade_schema(engine, auto=False)
    assert upgrade_schema(engine) == '0004'
    assert 'institute' in {col['name'] for col in inspect(engine).get_columns('documents')}
    with factory() as db:
        doc = db.get(Document, 'a' * 32)
        assert doc.title == 'Original' and doc.institute is None


def test_institutional_access_audit_requires_admin_for_export(sandbox):
    api, factory = sandbox
    registered = put(api, 'ligno', 'Fibralfa 45 MPa', institute='inovaligno')
    doc = registered.json()['id']
    assert api.get(f'/v1/documents/{doc}', headers=auth('ligno')).status_code == 200
    assert api.get(f'/v1/documents/{doc}/passages', headers=auth('peer')).status_code == 200
    assert api.get('/v1/search?q=fibralfa', headers=auth('ligno')).status_code == 200
    assert api.get('/v1/audit/events', headers=auth('poly')).status_code == 403
    assert api.get('/v1/audit/events').status_code == 401
    audit_response = api.get('/v1/audit/events?actor=ligno', headers=auth('admin'))
    assert audit_response.status_code == 200
    actions = {record['action'] for record in audit_response.json()['items']}
    assert 'document.metadata.read' in actions and 'document.search.read' in actions
    assert 'document.upload' in actions
    assert all('lig-ligno' not in str(record) for record in audit_response.json()['items'])
    other = api.get('/v1/audit/events?actor=peer', headers=auth('admin')).json()
    assert any(r['action'] == 'document.passages.read' for r in other['items'])
