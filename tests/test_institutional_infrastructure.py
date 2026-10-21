"""BIA-02.6: migrações, snapshots append-only e protocolo por instituto."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError, DatabaseError

from apps.api.main import make_app
from bia_core.db import (Base, Document, DatasetSnapshot, InstituteAssessment,
                         make_session_factory)
from bia_core.schema import upgrade_schema, VERSION_TABLE
from bia_core.institutional import institute_report

KEYS = {
    name: {"actor": name, "role": "admin" if name == "admin" else "curator",
           "institutes": institutes}
    for name, institutes in {
        "alice": [], "bruno": [], "carla": ["inovaligno"],
        "diana": ["polissacarideos"], "enzo": ["circularidade"], "admin": [],
    }.items()
}
KEYS["reader"] = {"actor": "reader", "role": "viewer"}


def auth(name):
    return {"Authorization": f"Bearer {name}"}


def sample_pdf(text_value="Celulose reforca polimeros com 32 MPa"):
    result = BytesIO()
    page = canvas.Canvas(result, invariant=1)
    page.drawString(40, 650, text_value)
    page.save()
    return result.getvalue()


@pytest.fixture()
def environment(tmp_path):
    url = f"sqlite:///{tmp_path / 'bia.sqlite3'}"
    client = TestClient(make_app(db_url=url, storage_dir=tmp_path / 'files',
                                 artifact_root=tmp_path / 'models', keys=KEYS))
    return client, make_session_factory(url), url


def populated(client, *, name="Celulose reforca polimeros com 32 MPa"):
    created = client.post('/v1/documents', headers=auth('alice'),
                          data={'title': name, 'rights_basis': 'open_access', 'training_allowed': 'true'},
                          files={'file': ('source.pdf', sample_pdf(name), 'application/pdf')})
    assert created.status_code == 201, created.text
    doc = created.json()['id']
    passage = client.get(f'/v1/documents/{doc}/passages', headers=auth('alice')).json()[0]['id']
    assert client.post(f'/v1/passages/{passage}/submit', headers=auth('alice')).status_code == 200
    assert client.post(f'/v1/passages/{passage}/certify', headers=auth('bruno')).status_code == 200
    return doc, passage


def completed_task(client, pid, *, label_a='NONE', label_b='NONE'):
    created = client.post('/v1/agreement/tasks', headers=auth('admin'), json={
        'passage_id': pid, 'start': 0, 'end': 8, 'rater_a': 'alice', 'rater_b': 'bruno'})
    assert created.status_code == 201, created.text
    task_id = created.json()['id']
    for who, label in [('alice', label_a), ('bruno', label_b)]:
        response = client.post(f'/v1/agreement/tasks/{task_id}/vote', headers=auth(who), json={'label': label})
        assert response.status_code == 200, response.text
    return task_id


def test_schema_migrates_old_versionless_db_and_preserves_documents(tmp_path):
    db = f"sqlite:///{tmp_path / 'legacy.sqlite3'}"
    engine = create_engine(db)
    old = [table for table in Base.metadata.sorted_tables
           if table.name not in {'dataset_snapshots', 'institute_assessments'}]
    Base.metadata.create_all(engine, tables=old)
    with engine.begin() as conn:
        conn.execute(Document.__table__.insert().values(
            id='a' * 32, title='LEGACY', sha256='b' * 64, rights_basis='open_access',
            training_allowed=True, asset_path='archive.pdf', status='ready'))
    assert 'bia_schema_versions' not in inspect(engine).get_table_names()
    factory = make_session_factory(db)
    with factory() as session:
        assert session.get(Document, 'a' * 32).title == 'LEGACY'
        assert session.execute(text('SELECT version FROM bia_schema_versions ORDER BY version')).scalars().all() == ['0001', '0002']
    assert {'dataset_snapshots', 'institute_assessments'}.issubset(inspect(engine).get_table_names())
    assert upgrade_schema(engine) == '0002'


def test_schema_requires_explicit_upgrade_and_rejects_partial(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'empty.sqlite3'}")
    with pytest.raises(RuntimeError, match='sem versão'):
        upgrade_schema(engine, auto=False)
    assert VERSION_TABLE not in inspect(engine).get_table_names()
    with engine.begin() as conn:
        conn.exec_driver_sql('CREATE TABLE documents (id TEXT PRIMARY KEY)')
    with pytest.raises(RuntimeError, match='parcial'):
        upgrade_schema(engine)
    with engine.connect() as conn:
        assert 'bia_schema_versions' not in inspect(conn).get_table_names()


def test_missing_immutability_trigger_is_rejected(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite3'}")
    upgrade_schema(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql('DROP TRIGGER bia_snapshot_no_delete')
    with pytest.raises(RuntimeError, match='Integridade'):
        upgrade_schema(engine)


def test_snapshot_idempotent_immutable_and_hash_verified(environment):
    client, factory, _ = environment
    doc, pid = populated(client)
    unauthenticated = client.post('/v1/datasets/gold/snapshots')
    assert unauthenticated.status_code == 401
    assert client.post('/v1/datasets/gold/snapshots', headers=auth('reader')).status_code == 403
    first = client.post('/v1/datasets/gold/snapshots', headers=auth('admin'))
    assert first.status_code == 201, first.text
    digest = first.json()['sha256']
    assert first.json()['created']
    duplicate = client.post('/v1/datasets/gold/snapshots', headers=auth('admin'))
    assert duplicate.status_code == 201 and not duplicate.json()['created']
    assert duplicate.json()['sha256'] == digest
    assert client.get('/v1/datasets/gold/snapshots', headers=auth('reader')).status_code == 403
    archive = client.get(f'/v1/datasets/gold/snapshots/{digest}', headers=auth('admin'))
    assert archive.status_code == 200 and archive.json()['sha256'] == digest
    with factory() as session:
        assert len(session.scalars(select(DatasetSnapshot)).all()) == 1
        row = session.get(DatasetSnapshot, digest)
        assert row.payload_json.startswith('{')
    engine = factory.kw['bind']
    with pytest.raises(DatabaseError):
        with engine.begin() as conn:
            conn.execute(text('UPDATE dataset_snapshots SET created_by = :name WHERE sha256=:sha'),
                         {'name': 'attacker', 'sha': digest})
    with pytest.raises(DatabaseError):
        with engine.begin() as conn:
            conn.execute(text('DELETE FROM dataset_snapshots WHERE sha256 = :sha'), {'sha': digest})
    assert client.get(f'/v1/datasets/gold/snapshots/{digest}', headers=auth('admin')).status_code == 200


def test_snapshot_rights_revoked_block_read_of_old_text(environment):
    client, factory, _ = environment
    doc, _ = populated(client)
    digest = client.post('/v1/datasets/gold/snapshots', headers=auth('admin')).json()['sha256']
    with factory.begin() as session:
        session.get(Document, doc).training_allowed = False
    assert client.get(f'/v1/datasets/gold/snapshots/{digest}', headers=auth('admin')).status_code == 403
    assert client.post('/v1/datasets/gold/snapshots', headers=auth('admin')).status_code == 409


def test_institute_membership_blind_protocol_independent_review_and_report(environment):
    client, factory, _ = environment
    _, pid = populated(client)
    tid = completed_task(client, pid)
    proto = client.get('/v1/institutes/inovaligno/protocol', headers=auth('carla'))
    assert proto.status_code == 200 and proto.json()['blind_to_gold']
    assert client.get('/v1/institutes/inovaligno/protocol', headers=auth('diana')).status_code == 403
    assert client.get('/v1/institutes/unknown/protocol', headers=auth('carla')).status_code == 404
    payload = {'task_id': tid, 'label': 'NONE'}
    assert client.post('/v1/institutes/inovaligno/assessments', headers=auth('alice'), json=payload).status_code == 403
    assert client.post('/v1/institutes/inovaligno/assessments', headers=auth('diana'), json=payload).status_code == 403
    a = client.post('/v1/institutes/inovaligno/assessments', headers=auth('carla'), json=payload)
    assert a.status_code == 201, a.text
    assert 'final_label' not in a.json()
    assert client.post('/v1/institutes/inovaligno/assessments', headers=auth('carla'), json=payload).status_code == 409
    wrong = client.post('/v1/institutes/polissacarideos/assessments', headers=auth('diana'),
                        json={'task_id': tid, 'label': 'MATERIAL'})
    assert wrong.status_code == 201, wrong.text
    assert client.get('/v1/institutes/assessments/report', headers=auth('carla')).status_code == 403
    report = client.get('/v1/institutes/assessments/report', headers=auth('admin'))
    assert report.status_code == 200
    institutes = {r['institute']: r for r in report.json()['institutes']}
    assert institutes['inovaligno']['exact_agreement'] == 1
    assert institutes['polissacarideos']['exact_agreement'] == 0
    assert institutes['circularidade']['exact_agreement'] is None
    assert report.json()['scientifically_validated'] is False
    with factory() as session:
        assert report.json()['sha256'] == institute_report(session)['sha256']
    with factory() as session:
        assert len(session.scalars(select(InstituteAssessment)).all()) == 2
    assert client.post(f'/v1/passages/{pid}/reopen', headers=auth('carla')).status_code == 200
    refreshed = client.get('/v1/institutes/assessments/report', headers=auth('admin')).json()
    assert all(row['assessed_tasks'] == 0 for row in refreshed['institutes'])
    assert client.post(f'/v1/passages/{pid}/submit', headers=auth('alice')).status_code == 200
    assert client.post(f'/v1/passages/{pid}/certify', headers=auth('bruno')).status_code == 200
    assert client.post('/v1/institutes/inovaligno/assessments', headers=auth('carla'), json=payload).status_code == 201
    refreshed = client.get('/v1/institutes/assessments/report', headers=auth('admin')).json()
    assert refreshed['institutes'][0]['assessed_tasks'] == 1


def test_institute_cannot_evaluate_incomplete_gold(environment):
    client, _, _ = environment
    _, pid = populated(client)
    created = client.post('/v1/agreement/tasks', headers=auth('admin'), json={
        'passage_id': pid, 'start': 0, 'end': 8, 'rater_a': 'alice', 'rater_b': 'bruno'})
    tid = created.json()['id']
    response = client.post('/v1/institutes/inovaligno/assessments', headers=auth('carla'),
                           json={'task_id': tid, 'label': 'NONE'})
    assert response.status_code == 409


def test_concurrent_votes_single_committed_row(environment):
    client, factory, _ = environment
    _, pid = populated(client)
    tid = client.post('/v1/agreement/tasks', headers=auth('admin'), json={
        'passage_id': pid, 'start': 0, 'end': 8, 'rater_a': 'alice', 'rater_b': 'bruno'}).json()['id']
    def submit():
        return client.post(f'/v1/agreement/tasks/{tid}/vote', headers=auth('alice'), json={'label': 'NONE'}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: submit(), range(2)))
    assert sorted(results) == [200, 409]
    with factory() as session:
        from bia_core.db import AgreementVote
        assert len(session.scalars(select(AgreementVote)).all()) == 1
