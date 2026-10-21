"""BIA-02.7: state transitions using single conditional SQL statements.

Rows affected are the concurrency decision: stale requests must not silently
overwrite certification state. The DB uniqueness constraint serializes first submit.
"""
from __future__ import annotations

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import PassageReview, now


def submit_once(session, passage_id: str, actor: str) -> bool:
    dialect = session.get_bind().dialect.name
    stmt = (postgres_insert(PassageReview) if dialect == 'postgresql' else
            sqlite_insert(PassageReview) if dialect == 'sqlite' else None)
    if stmt is None:
        raise RuntimeError('Transições condicionais não homologadas para este banco')
    created = session.execute(stmt.values(
        passage_id=passage_id, status='submitted', submitted_by=actor,
        submitted_at=now()).on_conflict_do_nothing(index_elements=['passage_id']))
    if created.rowcount == 1:
        session.expire_all()
        return True
    result = session.execute(update(PassageReview).where(
        PassageReview.passage_id == passage_id,
        PassageReview.status == 'draft',
    ).values(status='submitted', submitted_by=actor, submitted_at=now(),
             certified_by=None, certified_at=None))
    session.expire_all()
    return result.rowcount == 1


def certify_once(session, passage_id: str, actor: str) -> bool:
    result = session.execute(update(PassageReview).where(
        PassageReview.passage_id == passage_id,
        PassageReview.status == 'submitted',
        PassageReview.submitted_by != actor,
    ).values(status='certified', certified_by=actor, certified_at=now()))
    session.expire_all()
    return result.rowcount == 1


def reopen_once(session, passage_id: str) -> bool:
    result = session.execute(update(PassageReview).where(
        PassageReview.passage_id == passage_id,
        PassageReview.status.in_(('submitted', 'certified')),
    ).values(status='draft', certified_by=None, certified_at=None))
    session.expire_all()
    return result.rowcount == 1
