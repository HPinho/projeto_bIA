"""Invariantes de adjudicação independente com evidência e offsets preservados."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from .db import Annotation, AnnotationDispute, Passage
from .model import ENTITY_LABELS


def unresolved_count(session: Session, passage_id: str) -> int:
    return len(session.scalars(select(AnnotationDispute.id).where(
        AnnotationDispute.passage_id == passage_id,
        AnnotationDispute.status != "resolved"
    )).all())


def validate_alternative(passage: Passage, label: str, start: int, end: int) -> None:
    if label not in ENTITY_LABELS or not (0 <= start < end <= len(passage.text)):
        raise ValueError("Alternativa deve conter entidade e offsets válidos do texto original")
    if not passage.text[start:end].strip():
        raise ValueError("Alternativa vazia não representa uma entidade")


def has_overlap(session: Session, passage_id: str, start: int, end: int) -> bool:
    rows = session.scalars(select(Annotation).where(
        Annotation.passage_id == passage_id, Annotation.status != "rejected"
    )).all()
    return any(start < a.end and a.start < end for a in rows)


def dispute_record(record: AnnotationDispute, passage: Passage) -> dict:
    def alternative(actor, label, start, end):
        return ({"actor": actor, "label": label, "start": start, "end": end,
                 "quote": passage.text[start:end]}
                if actor is not None and label is not None and start is not None and end is not None else None)
    return {"id": record.id, "passage_id": passage.id, "status": record.status,
            "first": alternative(record.first_actor, record.first_label, record.first_start, record.first_end),
            "second": alternative(record.second_actor, record.second_label, record.second_start, record.second_end),
            "adjudicated_by": record.adjudicated_by, "resolution": record.resolution,
            "reason": record.reason,
            "annotation_id": record.annotation_id}
