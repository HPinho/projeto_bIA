"""Persistência local ou PostgreSQL; todo dado científico inclui proveniência."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def new_id() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    doi: Mapped[str | None] = mapped_column(String(255), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rights_basis: Mapped[str] = mapped_column(String(64), nullable=False)
    training_allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    asset_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    passages: Mapped[list[Passage]] = relationship(cascade="all, delete-orphan")


class Passage(Base):
    __tablename__ = "passages"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), nullable=False, index=True)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_offset: Mapped[int] = mapped_column(Integer, default=0)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)


class Annotation(Base):
    __tablename__ = "annotations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    passage_id: Mapped[str] = mapped_column(ForeignKey("passages.id"), nullable=False, index=True)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    start: Mapped[int] = mapped_column(Integer, nullable=False)
    end: Mapped[int] = mapped_column(Integer, nullable=False)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    reviewer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PassageReview(Base):
    """Full-span adjudication: missing labels mean O only after certification.

    A separate table preserves compatibility with the existing bootstrap schema.
    """
    __tablename__ = "passage_reviews"
    passage_id: Mapped[str] = mapped_column(ForeignKey("passages.id"), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="submitted")
    submitted_by: Mapped[str] = mapped_column(String(128), nullable=False)
    certified_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    certified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def database_url() -> str:
    return os.getenv("BIA_DATABASE_URL", "sqlite:///./data/bia.sqlite3")


def make_session_factory(url: str | None = None):
    url = url or database_url()
    if url.startswith("sqlite:///" ) and ":memory:" not in url:
        from pathlib import Path
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite:") else {}, pool_pre_ping=True)
    Base.metadata.create_all(engine)  # bootstrap v0; Alembic migration required before production
    return sessionmaker(bind=engine, expire_on_commit=False)


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    artifact_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ScientificRelation(Base):
    """Relação anotada; a certificação do trecho é versionada por hash.

    Relações aprovadas não permanecem publicáveis após reabertura/recertificação
    da passagem, mesmo que a evidência textual ainda exista.
    """
    __tablename__ = "scientific_relations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    passage_id: Mapped[str] = mapped_column(ForeignKey("passages.id"), nullable=False, index=True)
    subject_annotation_id: Mapped[str] = mapped_column(ForeignKey("annotations.id"), nullable=False)
    object_annotation_id: Mapped[str] = mapped_column(ForeignKey("annotations.id"), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(48), nullable=False)
    evidence_start: Mapped[int] = mapped_column(Integer, nullable=False)
    evidence_end: Mapped[int] = mapped_column(Integer, nullable=False)
    certification_key: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    proposed_by: Mapped[str] = mapped_column(String(128), nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
