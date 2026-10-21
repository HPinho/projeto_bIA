"""Curadoria explícita de relações científicas, nunca inferir relação por proximidade.

Sugestões são somente pares de entidades do mesmo trecho certificado. Uma
relação é publicável APENAS com evidência literal, duas identidades distintas
(proponente/revisor), entidades aprovadas e certificação corrente.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Annotation, Document, Passage, PassageReview, ScientificRelation


# A direção é parte da semântica. Nenhum predicado implica causalidade.
RELATION_SIGNATURES: dict[str, tuple[str, str]] = {
    "MATERIAL_HAS_PROPERTY": ("MATERIAL", "PROPERTY"),
    "MATERIAL_PROCESSED_BY": ("MATERIAL", "PROCESS"),
    "PROCESS_HAS_CONDITION": ("PROCESS", "CONDITION"),
    "PROCESS_HAS_QUANTITY": ("PROCESS", "QUANTITY"),
    "PROPERTY_HAS_QUANTITY": ("PROPERTY", "QUANTITY"),
    "QUANTITY_HAS_UNIT": ("QUANTITY", "UNIT"),
    "MATERIAL_HAS_APPLICATION": ("MATERIAL", "APPLICATION"),
    "MATERIAL_CONTAINS": ("MATERIAL", "MATERIAL"),
}
MAX_EVIDENCE_CHARS = 1000
MAX_PAIR_DISTANCE = 300


def certification_key(review: PassageReview | None) -> str | None:
    """Identificador estável na leitura SQLite/PostgreSQL para uma certificação."""
    if review is None or review.status != "certified" or review.certified_at is None:
        return None
    # now() cria horários UTC; SQLite retorna datetimes sem tzinfo.
    timestamp = review.certified_at.replace(tzinfo=None).isoformat(timespec="microseconds")
    submitted = review.submitted_at.replace(tzinfo=None).isoformat(timespec="microseconds")
    payload = f"{review.passage_id}\0{submitted}\0{timestamp}\0{review.submitted_by}\0{review.certified_by}"
    return sha256(payload.encode("utf-8")).hexdigest()


def certified_anchors(session: Session, passage_id: str) -> tuple[str | None, dict[str, Annotation]]:
    current = certification_key(session.get(PassageReview, passage_id))
    if not current:
        return None, {}
    accepted = session.scalars(select(Annotation).where(
        Annotation.passage_id == passage_id, Annotation.status == "approved"
    )).all()
    return current, {ann.id: ann for ann in accepted}


def is_current(session: Session, rel: ScientificRelation) -> bool:
    current, approved = certified_anchors(session, rel.passage_id)
    if rel.status != "approved" or not current or current != rel.certification_key:
        return False
    source = approved.get(rel.subject_annotation_id)
    target = approved.get(rel.object_annotation_id)
    if not source or not target:
        return False
    return (source.label, target.label) == RELATION_SIGNATURES.get(rel.relation_type)


def candidate_pairs(session: Session, passage_id: str) -> list[dict]:
    """Heurística lexical SEM declarar que uma relação seja verdadeira."""
    passage = session.get(Passage, passage_id)
    if not passage:
        return []
    key, approved = certified_anchors(session, passage_id)
    if not key:
        return []
    entities = sorted(approved.values(), key=lambda a: (a.start, a.end, a.id))
    result = []
    for left, right in combinations(entities, 2):
        begin, end = min(left.start, right.start), max(left.end, right.end)
        if end - begin > MAX_PAIR_DISTANCE:
            continue
        for name, (subject_kind, object_kind) in RELATION_SIGNATURES.items():
            possible = []
            if (left.label, right.label) == (subject_kind, object_kind):
                possible.append((left, right))
            if (left.label, right.label) == (object_kind, subject_kind) and left.id != right.id:
                # Relações MATERIAL_CONTAINS são direcionais: ambos os sentidos
                # devem permanecer apenas hipóteses até a avaliação humana.
                if left.label != right.label:
                    possible.append((right, left))
                elif name == "MATERIAL_CONTAINS":
                    possible.append((right, left))
            for subject, obj in possible:
                result.append({
                    "subject_annotation_id": subject.id,
                    "object_annotation_id": obj.id,
                    "relation_type": name,
                    "evidence_start": begin, "evidence_end": end,
                    "evidence_quote": passage.text[begin:end],
                    "status": "candidate_unverified",
                })
    return result


def relation_record(rel: ScientificRelation, passage: Passage, document: Document, anchors: dict[str, Annotation]) -> dict:
    source, target = anchors[rel.subject_annotation_id], anchors[rel.object_annotation_id]
    return {
        "id": rel.id, "relation_type": rel.relation_type, "status": "verified",
        "document_id": document.id, "title": document.title, "doi": document.doi,
        "rights_basis": document.rights_basis, "page": passage.page, "passage_id": passage.id,
        "subject": {"annotation_id": source.id, "label": source.label, "start": source.start,
                    "end": source.end, "quote": passage.text[source.start:source.end]},
        "object": {"annotation_id": target.id, "label": target.label, "start": target.start,
                   "end": target.end, "quote": passage.text[target.start:target.end]},
        "evidence": {"start": rel.evidence_start, "end": rel.evidence_end,
                     "quote": passage.text[rel.evidence_start:rel.evidence_end]},
        "review": {"proposed_by": rel.proposed_by, "reviewed_by": rel.reviewed_by},
    }
