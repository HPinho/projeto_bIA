"""BIA-02.7: custodial retention and corpus-readiness safeguards.

Retention deadlines trigger manual review, NEVER automatic deletion. Restricted
sources remain available only for administrative audit, and training revocation
is irreversible through this endpoint (regrant needs separate rights validation).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone, timedelta

from sqlalchemy import select

from .db import Document, DocumentGovernance, Passage, PassageReview, now


def set_governance(session, doc: Document, *, actor: str, status: str,
                   retention_until: datetime | None, legal_hold: bool,
                   reason: str) -> DocumentGovernance:
    row = session.get(DocumentGovernance, doc.id)
    if row is None:
        row = DocumentGovernance(document_id=doc.id, reason=reason, updated_by=actor,
                                 access_status='active')
        session.add(row)
    if status == 'restricted':
        if doc.status != 'restricted':
            row.original_status = doc.status
        doc.status = 'restricted'
        doc.training_allowed = False
    elif doc.status == 'restricted':
        # Restoring visibility NEVER restores the prior training authorization.
        doc.status = row.original_status or 'needs_review'
    row.access_status = status
    row.retention_until = retention_until.astimezone(timezone.utc) if retention_until else None
    row.legal_hold = legal_hold
    row.reason = reason.strip()
    row.updated_by = actor
    row.updated_at = now()
    session.flush()
    return row


def governance_payload(doc: Document, row: DocumentGovernance | None) -> dict:
    return {
        'document_id': doc.id,
        'access_status': 'restricted' if doc.status == 'restricted' else 'active',
        'status': doc.status,
        'training_allowed': doc.training_allowed,
        'legal_hold': bool(row.legal_hold) if row else False,
        'retention_until': row.retention_until.isoformat() if row and row.retention_until else None,
        'reason': row.reason if row else None,
        'updated_by': row.updated_by if row else None,
        'physical_deletion_supported': False,
    }


def governance_report(session) -> dict:
    """Aggregate-only rights inventory, not a validation of training quality."""
    docs = session.scalars(select(Document)).all()
    controls = {r.document_id: r for r in session.scalars(select(DocumentGovernance)).all()}
    certified = {pid for (pid,) in session.execute(
        select(PassageReview.passage_id).where(PassageReview.status == 'certified')).all()}
    passages = session.execute(select(Passage.document_id, Passage.id)).all()
    certified_by_doc = Counter(docid for docid, pid in passages if pid in certified)
    utc_now = datetime.now(timezone.utc)
    soon = utc_now + timedelta(days=90)
    due = 0
    approaching = 0
    held = 0
    for control in controls.values():
        held += int(control.legal_hold)
        if control.retention_until:
            deadline = control.retention_until.replace(tzinfo=timezone.utc) if control.retention_until.tzinfo is None else control.retention_until
            due += int(deadline <= utc_now)
            approaching += int(utc_now < deadline <= soon)
    counts = Counter(d.rights_basis for d in docs)
    return {
        'schema': 'bia.corpus_readiness.v1',
        'scientifically_validated': False,
        'documents': len(docs),
        'rights_basis': dict(sorted(counts.items())),
        'restricted_documents': sum(d.status == 'restricted' for d in docs),
        'training_authorized_ready_documents': sum(d.training_allowed and d.status == 'ready' for d in docs),
        'training_authorized_certified_passages': sum(certified_by_doc[d.id] for d in docs
                                                       if d.training_allowed and d.status == 'ready'),
        'legal_holds': held,
        'retention_review_due': due,
        'retention_review_due_within_90_days': approaching,
        'unreviewed_documents': sum(d.status == 'needs_review' for d in docs),
        'warning': 'Inventário documental, não um corpus validado. Nenhuma exclusão automática é realizada.',
    }


def corpus_manifest(session, *, offset: int = 0, limit: int = 100) -> dict:
    """Admin-only, paginated metadata inventory; no PDF/text/paths exported."""
    docs = session.scalars(select(Document).order_by(Document.id).offset(offset).limit(limit)).all()
    if not docs:
        return {'items': [], 'offset': offset, 'limit': limit, 'scientifically_validated': False}
    ids = [doc.id for doc in docs]
    certified = Counter(doc_id for (doc_id,) in session.execute(
        select(Passage.document_id).join(PassageReview, PassageReview.passage_id == Passage.id)
        .where(Passage.document_id.in_(ids), PassageReview.status == 'certified')).all())
    result = []
    for doc in docs:
        issues = []
        if doc.status == 'restricted':
            issues.append('access_restricted')
        if not doc.training_allowed:
            issues.append('training_not_authorized')
        if doc.status == 'needs_review':
            issues.append('extraction_review_required')
        if not certified[doc.id]:
            issues.append('no_certified_passages')
        result.append({'id': doc.id, 'title': doc.title, 'doi': doc.doi, 'year': doc.year,
                       'sha256': doc.sha256, 'status': doc.status, 'rights_basis': doc.rights_basis,
                       'training_allowed': doc.training_allowed,
                       'certified_passages': certified[doc.id], 'readiness_issues': issues})
    return {'items': result, 'offset': offset, 'limit': limit, 'scientifically_validated': False,
            'warning': 'Inventário interno para revisão humana; não comprova qualidade científica ou permissões de redistribuição.'}
