"""Curation rules used by HTTP endpoints and the local training pipeline."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import Annotation, PassageReview
from .adjudication import unresolved_count


def annotations_for(session: Session, passage_id: str) -> list[Annotation]:
    return list(session.scalars(
        select(Annotation).where(Annotation.passage_id == passage_id)
        .order_by(Annotation.start, Annotation.end, Annotation.id)
    ))


def is_certified(session: Session, passage_id: str) -> bool:
    review = session.get(PassageReview, passage_id)
    return review is not None and review.status == "certified"


def review_state(session: Session, passage_id: str) -> dict:
    review = session.get(PassageReview, passage_id)
    annotations = annotations_for(session, passage_id)
    return {
        "status": review.status if review else "draft",
        "submitted_by": review.submitted_by if review and review.status != "draft" else None,
        "certified_by": review.certified_by if review and review.status == "certified" else None,
        "approved_annotations": sum(a.status == "approved" for a in annotations),
        "pending_annotations": sum(a.status == "pending" for a in annotations),
        "rejected_annotations": sum(a.status == "rejected" for a in annotations),
        "unresolved_disputes": unresolved_count(session, passage_id),
    }
