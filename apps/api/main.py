"""API científica bIA. Sem rede de IA, sem downloads de modelos externos."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from bia_core.db import Annotation, AuditEvent, Document, Passage, PassageReview, ScientificRelation, TrainingRun, RelationModelRun, RelationNegativeAssessment, make_session_factory, now
from bia_core.access import compile_keys, authenticate, can_read, upload_scope
from bia_core.ingestion import InvalidPDF, pages_from_pdf
from bia_core.curation import annotations_for, review_state
from bia_core.model import ENTITY_LABELS
from bia_core.retrieval import bm25
from bia_core.relations import (MAX_EVIDENCE_CHARS, RELATION_SIGNATURES, candidate_pairs,
                                certified_anchors, is_current, relation_record, certification_key)
from bia_core.training import extract, train
from bia_core.relation_learning import train_relations, predict_candidates
from bia_core.evaluation import verified_report
from bia_core.adjudication import validate_alternative, has_overlap, unresolved_count, dispute_record
from bia_core.bibliography import normalize_doi, signals, ordered_pair, publication_clusters
from bia_core.db import AnnotationDispute, BibliographicDecision, AgreementTask, AgreementVote, AgreementResolution, DatasetSnapshot, InstituteAssessment, DocumentGovernance
from bia_core.institutional import INSTITUTES, create_snapshot, current_snapshot, institute_report
from bia_core.transitions import submit_once, certify_once, reopen_once
from bia_core.governance import governance_report, governance_payload, set_governance, corpus_manifest
from bia_core.agreement import AGREEMENT_LABELS, agreement_report, task_state, votes_for, gold_snapshot, passage_agreement_ready

ROLES = ("viewer", "researcher", "curator", "admin")
ALLOWED_RIGHTS = ("open_access", "author_permission", "institutional_permission")


class NewAnnotation(BaseModel):
    passage_id: str
    label: Literal["MATERIAL", "PROCESS", "PROPERTY", "QUANTITY", "UNIT", "CONDITION", "APPLICATION"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]


class RelationReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]
    explicit_no_relation: bool = False


class RelationProposal(BaseModel):
    passage_id: str
    subject_annotation_id: str
    object_annotation_id: str
    relation_type: Literal[
        "MATERIAL_HAS_PROPERTY", "MATERIAL_PROCESSED_BY", "PROCESS_HAS_CONDITION",
        "PROCESS_HAS_QUANTITY", "PROPERTY_HAS_QUANTITY", "QUANTITY_HAS_UNIT",
        "MATERIAL_HAS_APPLICATION", "MATERIAL_CONTAINS",
    ]
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(gt=0)


class Alternative(BaseModel):
    label: Literal["MATERIAL", "PROCESS", "PROPERTY", "QUANTITY", "UNIT", "CONDITION", "APPLICATION"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class OpenDispute(BaseModel):
    passage_id: str
    alternative: Alternative


class ResolveDispute(BaseModel):
    resolution: Literal["first", "second", "neither"]
    reason: str = Field(min_length=10, max_length=500)


class BibliographyVerdict(BaseModel):
    left_document_id: str
    right_document_id: str
    verdict: Literal["same_publication", "distinct_publication"]
    reason: str = Field(min_length=10, max_length=500)


class NewAgreementTask(BaseModel):
    passage_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    rater_a: str = Field(min_length=1, max_length=128)
    rater_b: str = Field(min_length=1, max_length=128)


class AgreementLabel(BaseModel):
    label: Literal["MATERIAL", "PROCESS", "PROPERTY", "QUANTITY", "UNIT", "CONDITION", "APPLICATION", "NONE"]


class AgreementAdjudication(AgreementLabel):
    reason: str = Field(min_length=10, max_length=500)


class InstituteEvaluation(BaseModel):
    task_id: str
    label: Literal["MATERIAL", "PROCESS", "PROPERTY", "QUANTITY", "UNIT", "CONDITION", "APPLICATION", "NONE"]


class GovernanceDecision(BaseModel):
    access_status: Literal["active", "restricted"]
    retention_until: datetime | None = None
    legal_hold: bool = False
    release_hold: bool = False
    reason: str = Field(min_length=10, max_length=500)


class AssignInstitution(BaseModel):
    institute: Literal["inovaligno", "polissacarideos", "circularidade"]
    reason: str = Field(min_length=10, max_length=500)


class TrainRequest(BaseModel):
    epochs: int = Field(default=2, ge=1, le=25)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


def make_app(*, db_url: str | None = None, storage_dir: Path | None = None, keys: dict | None = None, artifact_root: Path | None = None) -> FastAPI:
    session_factory = make_session_factory(db_url)
    storage = Path(storage_dir or os.getenv("BIA_STORAGE_DIR", "./data/documents")).resolve()
    root = Path(artifact_root or os.getenv("BIA_MODEL_BASE_DIR", "./checkpoints")).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    key_config = keys if keys is not None else json.loads(os.getenv("BIA_API_KEYS", "{}"))
    compiled_keys = compile_keys(key_config)
    configured_curators = {profile["actor"] for _, profile in compiled_keys
                           if profile["role"] in ("curator", "admin")}
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bia-train")
    job_lock = Lock()
    app = FastAPI(title="bIA Scientific Core", version="0.1.0", docs_url="/docs")
    origins = os.getenv("BIA_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o for o in origins if o], allow_methods=["GET", "POST", "PUT"], allow_headers=["Authorization", "Content-Type"])

    def db():
        with session_factory() as session:
            yield session

    def principal(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Token Bearer obrigatório")
        token = authorization.removeprefix("Bearer ")
        if not token or len(token) > 4096:
            raise HTTPException(401, "Token inválido")
        profile = authenticate(compiled_keys, token)
        if profile:
            return profile
        raise HTTPException(401, "Token inválido")

    def role(*allowed):
        def require(user: dict = Depends(principal)):
            if user["role"] not in allowed:
                raise HTTPException(403, "Ação não autorizada")
            return user
        return require

    def visible(doc: Document, user: dict) -> bool:
        return can_read(doc, user)

    def audit(session, user, action, resource):
        session.add(AuditEvent(actor=user["actor"], action=action, resource=resource))

    def audit_sensitive_read(session, user, doc, action):
        if doc.rights_basis == "institutional_permission":
            audit(session, user, action, f"{doc.id}:{doc.institute or 'unclassified'}")
            session.commit()

    def flush_unique(session, message):
        try:
            session.flush()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(409, message) from exc

    def commit_unique(session, message):
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(409, message) from exc

    def lock_passage(session, passage_id):
        # PostgreSQL: lock the passage before edits, submit, certify or reopen.
        if session.get_bind().dialect.name == "postgresql":
            session.scalar(select(Passage).where(Passage.id == passage_id).with_for_update())

    def get_doc(session, doc_id, user):
        doc = session.get(Document, doc_id)
        if doc is None or not visible(doc, user):
            raise HTTPException(404, "Documento não encontrado")
        return doc

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "bia", "ai_provider": "local-only"}

    @app.get("/v1/auth/me")
    def whoami(user: dict = Depends(principal)):
        return {"actor": user["actor"], "role": user["role"],
                "institutes": list(user["institutes"]), "auth_method": "bearer-local"}

    @app.post("/v1/documents", status_code=201)
    async def upload_document(
        file: UploadFile = File(...), title: str = Form(...), rights_basis: str = Form(...),
        training_allowed: bool = Form(False), doi: str | None = Form(None), year: int | None = Form(None),
        institute: str | None = Form(None),
        user: dict = Depends(role("curator", "admin")), session: Session = Depends(db),
    ):
        if rights_basis not in ALLOWED_RIGHTS:
            raise HTTPException(422, "Defina uma base de direitos permitida")
        try:
            assigned_institute = upload_scope(user, rights_basis, institute)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if not 1 <= len(title.strip()) <= 500 or year is not None and not 1500 <= year <= 2100:
            raise HTTPException(422, "Título ou ano inválido")
        if doi and normalize_doi(doi) is None:
            raise HTTPException(422, "DOI inválido; use o formato 10.xxxx/sufixo")
        maximum = int(os.getenv("BIA_MAX_PDF_BYTES", "20971520"))
        if maximum <= 0 or maximum > 100 * 1024 * 1024:
            raise HTTPException(500, "Limite de upload mal configurado")
        contents = await file.read(maximum + 1)
        if len(contents) > maximum:
            raise HTTPException(413, "PDF excede o limite de upload")
        try:
            pages = pages_from_pdf(contents)
        except InvalidPDF as exc:
            raise HTTPException(422, str(exc)) from exc
        digest = sha256(contents).hexdigest()
        prior = session.scalar(select(Document).where(Document.sha256 == digest))
        if prior:
            # Sem revelar ID/metadata de documentos de outro instituto.
            if not visible(prior, user):
                raise HTTPException(409, "PDF já cadastrado")
            raise HTTPException(409, {"message": "PDF já cadastrado", "document_id": prior.id})
        document_path = storage / f"{digest}.pdf"
        # No uploaded filename is ever used for filesystem paths.
        document_path.write_bytes(contents)
        document = Document(title=title.strip(), doi=normalize_doi(doi) if doi else None, year=year,
                            institute=assigned_institute,
                            rights_basis=rights_basis, training_allowed=training_allowed,
                            sha256=digest, asset_path=str(document_path),
                            status="ready" if any(text.strip() for _, text in pages) else "needs_review")
        session.add(document)
        session.flush()
        for page, text in pages:
            if text.strip():
                session.add(Passage(document_id=document.id, page=page, text=text, start_offset=0, end_offset=len(text)))
        audit(session, user, "document.upload", document.id)
        session.commit()
        return {"id": document.id, "sha256": digest, "status": document.status,
                "institute": document.institute, "pages": len(pages)}

    @app.put("/v1/documents/{document_id}/governance")
    def update_document_governance(document_id: str, data: GovernanceDecision,
                                   user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "Documento não encontrado")
        if data.retention_until is not None and data.retention_until.tzinfo is None:
            raise HTTPException(422, "Prazo de retenção precisa incluir fuso horário")
        # PostgreSQL serializa decisões concorrentes sobre o mesmo documento.
        if session.get_bind().dialect.name == "postgresql":
            session.execute(select(Document).where(Document.id == document_id).with_for_update()).scalar_one()
            session.refresh(doc)
        previous = session.get(DocumentGovernance, document_id)
        if previous is not None and previous.legal_hold and not data.legal_hold and not data.release_hold:
            raise HTTPException(409, "Suspensão da guarda exige liberação explícita")
        if data.release_hold and (previous is None or not previous.legal_hold or data.legal_hold):
            raise HTTPException(409, "Liberação de guarda inconsistente")
        try:
            record = set_governance(session, doc, actor=user["actor"], status=data.access_status,
                                    retention_until=data.retention_until, legal_hold=data.legal_hold, reason=data.reason)
        except IntegrityError as exc:
            session.rollback()
            raise HTTPException(409, "Governança atualizada em outra transação") from exc
        audit(session, user, "document.governance.update", document_id)
        session.commit()
        return governance_payload(doc, record)

    @app.get("/v1/documents/{document_id}/governance")
    def get_document_governance(document_id: str, user: dict = Depends(role("admin",)),
                                session: Session = Depends(db)):
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "Documento não encontrado")
        return governance_payload(doc, session.get(DocumentGovernance, document_id))

    @app.get("/v1/corpus/readiness")
    def corpus_readiness(user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        return governance_report(session)

    @app.get("/v1/corpus/manifest")
    def corpus_documents(offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=100),
                         user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        return corpus_manifest(session, offset=offset, limit=limit)

    @app.put("/v1/documents/{document_id}/institution")
    def assign_document_institute(document_id: str, data: AssignInstitution,
                                  user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        doc = session.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "Documento não encontrado")
        if doc.rights_basis != "institutional_permission":
            raise HTTPException(409, "Somente documentos institucionais podem receber instituto")
        if session.get_bind().dialect.name == "postgresql":
            session.scalar(select(Document).where(Document.id == document_id).with_for_update())
            session.refresh(doc)
        if doc.institute and doc.institute != data.institute:
            raise HTTPException(409, "Transferência interinstitucional exige procedimento próprio")
        if doc.institute == data.institute:
            raise HTTPException(409, "Documento já classificado neste instituto")
        doc.institute = data.institute
        audit(session, user, "document.institution.assign", f"{document_id}:{data.institute}")
        session.commit()
        return {"document_id": doc.id, "institute": doc.institute, "status": "assigned"}

    @app.get("/v1/documents/{document_id}")
    def document_details(document_id: str, user: dict = Depends(role(*ROLES)), session: Session = Depends(db)):
        doc = get_doc(session, document_id, user)
        audit_sensitive_read(session, user, doc, "document.metadata.read")
        return {"id": doc.id, "title": doc.title, "doi": doc.doi, "year": doc.year,
                "rights_basis": doc.rights_basis, "training_allowed": doc.training_allowed,
                "institute": doc.institute, "sha256": doc.sha256, "status": doc.status}

    @app.get("/v1/documents/{document_id}/passages")
    def document_passages(document_id: str, user: dict = Depends(role(*ROLES)), session: Session = Depends(db)):
        document = get_doc(session, document_id, user)
        audit_sensitive_read(session, user, document, "document.passages.read")
        items = session.scalars(select(Passage).where(Passage.document_id == document_id).order_by(Passage.page)).all()
        return [{"id": p.id, "document_id": p.document_id, "page": p.page, "text": p.text,
                 "start_offset": p.start_offset, "end_offset": p.end_offset,
                 "review": review_state(session, p.id)} for p in items]

    @app.get("/v1/search")
    def search(q: str = Query(min_length=2, max_length=300), year: int | None = None,
               limit: int = Query(default=10, ge=1, le=50), user: dict = Depends(role(*ROLES)), session: Session = Depends(db)):
        pairs = session.execute(select(Passage, Document).join(Document, Passage.document_id == Document.id)
                                .where(Document.status == "ready")).all()
        pairs = [(passage, doc) for passage, doc in pairs if visible(doc, user) and (year is None or doc.year == year)]
        scores = bm25(q, [p.text for p, _ in pairs])
        ranked = sorted(((scores[i], p, d) for i, (p, d) in enumerate(pairs) if scores[i] > 0),
                        key=lambda item: (-item[0], item[1].id))[:limit]
        for doc in {d.id: d for _, _, d in ranked}.values():
            if doc.rights_basis == "institutional_permission":
                audit(session, user, "document.search.read", f"{doc.id}:{doc.institute}")
        if any(d.rights_basis == "institutional_permission" for _, _, d in ranked):
            session.commit()
        return {"query": q, "results": [{"score": score, "document_id": doc.id, "title": doc.title,
                 "doi": doc.doi, "page": p.page, "passage_id": p.id, "quote": p.text[:1200],
                 "rights_basis": doc.rights_basis} for score, p, doc in ranked],
                "status": "ok" if ranked else "insufficient_evidence"}

    @app.post("/v1/annotations", status_code=201)
    def annotate(data: NewAnnotation, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, data.passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        lock_passage(session, passage.id)
        if not passage.text[data.start:data.end] or data.end > len(passage.text):
            raise HTTPException(422, "Offsets excedem texto original")
        state = review_state(session, passage.id)
        if state["status"] != "draft":
            raise HTTPException(409, "Trecho submetido ou certificado; reabra antes de modificar")
        existing = [a for a in annotations_for(session, passage.id) if a.status != "rejected"]
        if any(data.start < a.end and a.start < data.end for a in existing):
            raise HTTPException(409, "Anotação sobreposta; revise a existente")
        annotation = Annotation(passage_id=passage.id, label=data.label, start=data.start, end=data.end, actor=user["actor"])
        session.add(annotation)
        session.flush()
        audit(session, user, "annotation.create", annotation.id)
        session.commit()
        return {"id": annotation.id, "status": annotation.status, "quote": passage.text[data.start:data.end]}

    @app.get("/v1/annotations/{passage_id}")
    def get_annotations(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        items = annotations_for(session, passage_id)
        return [{"id": a.id, "label": a.label, "start": a.start, "end": a.end,
                 "quote": passage.text[a.start:a.end], "status": a.status, "actor": a.actor,
                 "reviewer": a.reviewer} for a in items]

    @app.post("/v1/annotations/{annotation_id}/review")
    def review(annotation_id: str, data: ReviewDecision, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        annotation = session.get(Annotation, annotation_id)
        if annotation is None:
            raise HTTPException(404, "Anotação não encontrada")
        passage = session.get(Passage, annotation.passage_id)
        get_doc(session, passage.document_id, user)
        lock_passage(session, passage.id)
        if review_state(session, annotation.passage_id)["status"] != "draft":
            raise HTTPException(409, "Trecho submetido ou certificado; reabra antes de modificar")
        if annotation.status != "pending":
            raise HTTPException(409, "Anotação já revisada")
        if annotation.actor == user["actor"]:
            raise HTTPException(403, "Revisão deve ser feita por outra pessoa")
        annotation.status = data.decision
        annotation.reviewer = user["actor"]
        annotation.reviewed_at = now()
        audit(session, user, "annotation.review." + data.decision, annotation.id)
        session.commit()
        return {"id": annotation.id, "status": annotation.status}

    @app.get("/v1/passages/{passage_id}/review")
    def get_passage_review(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        return review_state(session, passage_id)

    @app.get("/v1/audit/events")
    def audit_events(limit: int = Query(default=100, ge=1, le=500),
                     actor: str | None = Query(default=None, max_length=128),
                     user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        query = select(AuditEvent).order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
        if actor:
            query = query.where(AuditEvent.actor == actor)
        rows = session.scalars(query.limit(limit)).all()
        return {"items": [{"id": row.id, "actor": row.actor, "action": row.action,
                           "resource": row.resource, "created_at": row.created_at.isoformat()} for row in rows],
                "total_returned": len(rows), "audit_scope": "global_admin_only"}

    @app.get("/v1/curation/summary")
    def curation_summary(user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        """Readiness counts, never confuse annotation volume with validated science."""
        documents = [d for d in session.scalars(select(Document)).all() if visible(d, user)]
        visible_ids = {d.id for d in documents}
        reviews = {
            row.passage_id: row.status
            for row in session.scalars(select(PassageReview)).all()
        }
        passages = [p for p in session.scalars(select(Passage)).all() if p.document_id in visible_ids]
        states = {"draft": 0, "submitted": 0, "certified": 0}
        eligible = 0
        open_disputes = sum(1 for case in session.scalars(select(AnnotationDispute)).all()
                            if case.status != "resolved" and case.passage_id in {p.id for p in passages})
        bibliography_decisions = sum(1 for row in session.scalars(select(BibliographicDecision)).all()
                                     if row.left_document_id in visible_ids and row.right_document_id in visible_ids)
        allowed = {d.id for d in documents if d.training_allowed}
        for passage in passages:
            status = reviews.get(passage.id, "draft")
            states[status] += 1
            if status == "certified" and passage.document_id in allowed:
                eligible += 1
        return {
            "documents": len(documents), "training_authorized_documents": len(allowed),
            "passages": len(passages), "passages_by_state": states,
            "training_eligible_passages": eligible,
            "open_adjudications": open_disputes, "bibliographic_pairs_reviewed": bibliography_decisions,
            "scientifically_validated": False,
        }

    @app.post("/v1/passages/{passage_id}/submit")
    def submit_passage(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        lock_passage(session, passage.id)
        state = review_state(session, passage_id)
        if state["status"] != "draft":
            raise HTTPException(409, "Trecho já submetido ou certificado")
        if state["pending_annotations"] or state["unresolved_disputes"]:
            raise HTTPException(409, "Resolva anotações pendentes e divergências antes de submeter")
        if not submit_once(session, passage_id, user["actor"]):
            raise HTTPException(409, "Trecho alterado em transação concorrente")
        audit(session, user, "passage.review.submit", passage_id)
        session.commit()
        return review_state(session, passage_id)

    @app.post("/v1/passages/{passage_id}/certify")
    def certify_passage(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        lock_passage(session, passage.id)
        review = session.get(PassageReview, passage_id)
        if review is None or review.status != "submitted":
            raise HTTPException(409, "Trecho ainda não foi submetido")
        if review.submitted_by == user["actor"]:
            raise HTTPException(403, "Certificação exige um segundo pesquisador")
        state = review_state(session, passage_id)
        if state["pending_annotations"] or state["unresolved_disputes"]:
            raise HTTPException(409, "Existem anotações ou divergências pendentes")
        if not certify_once(session, passage_id, user["actor"]):
            raise HTTPException(409, "Certificação alterada em transação concorrente")
        audit(session, user, "passage.review.certify", passage_id)
        session.commit()
        return review_state(session, passage_id)

    @app.post("/v1/passages/{passage_id}/reopen")
    def reopen_passage(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        lock_passage(session, passage.id)
        review = session.get(PassageReview, passage_id)
        if review is None or review.status == "draft":
            raise HTTPException(409, "Trecho já está em rascunho")
        if not reopen_once(session, passage_id):
            raise HTTPException(409, "Trecho alterado em transação concorrente")
        audit(session, user, "passage.review.reopen", passage_id)
        session.commit()
        return review_state(session, passage_id)

    @app.post("/v1/adjudications", status_code=201)
    def open_dispute(data: OpenDispute, user: dict = Depends(role("curator", "admin")),
                     session: Session = Depends(db)):
        passage = session.get(Passage, data.passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        if review_state(session, passage.id)["status"] != "draft":
            raise HTTPException(409, "Reabra o trecho antes de registrar divergências")
        try:
            validate_alternative(passage, data.alternative.label, data.alternative.start, data.alternative.end)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        case = AnnotationDispute(passage_id=passage.id, first_actor=user["actor"],
            first_label=data.alternative.label, first_start=data.alternative.start,
            first_end=data.alternative.end)
        session.add(case)
        session.flush()
        audit(session, user, "adjudication.open", case.id)
        session.commit()
        return dispute_record(case, passage)

    @app.post("/v1/adjudications/{case_id}/second")
    def second_opinion(case_id: str, data: Alternative,
                       user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        case = session.get(AnnotationDispute, case_id)
        if case is None:
            raise HTTPException(404, "Divergência não encontrada")
        passage = session.get(Passage, case.passage_id)
        get_doc(session, passage.document_id, user)
        if case.status != "awaiting_second" or review_state(session, passage.id)["status"] != "draft":
            raise HTTPException(409, "Divergência não aceita segunda opinião")
        if user["actor"] == case.first_actor:
            raise HTTPException(403, "A segunda opinião exige outro pesquisador")
        try:
            validate_alternative(passage, data.label, data.start, data.end)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        case.second_actor = user["actor"]
        case.second_label, case.second_start, case.second_end = data.label, data.start, data.end
        case.status = "awaiting_adjudication"
        audit(session, user, "adjudication.second", case.id)
        session.commit()
        return dispute_record(case, passage)

    @app.post("/v1/adjudications/{case_id}/resolve")
    def resolve_dispute(case_id: str, data: ResolveDispute,
                        user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        case = session.get(AnnotationDispute, case_id)
        if case is None:
            raise HTTPException(404, "Divergência não encontrada")
        passage = session.get(Passage, case.passage_id)
        get_doc(session, passage.document_id, user)
        if case.status != "awaiting_adjudication" or review_state(session, passage.id)["status"] != "draft":
            raise HTTPException(409, "Divergência não está pronta para adjudicação")
        if user["actor"] in (case.first_actor, case.second_actor):
            raise HTTPException(403, "A adjudicação exige terceira identidade independente")
        if data.resolution != "neither":
            actor, label, start, end = ((case.first_actor, case.first_label, case.first_start, case.first_end)
                if data.resolution == "first" else
                (case.second_actor, case.second_label, case.second_start, case.second_end))
            if has_overlap(session, passage.id, start, end):
                raise HTTPException(409, "Anotação conflita com outra existente; resolva-a antes")
            selected = Annotation(passage_id=passage.id, label=label, start=start, end=end,
                                  actor=actor, status="approved", reviewer=user["actor"], reviewed_at=now())
            session.add(selected)
            session.flush()
            case.annotation_id = selected.id
        case.status = "resolved"
        case.resolution = data.resolution
        case.reason = data.reason.strip()
        case.adjudicated_by = user["actor"]
        case.resolved_at = now()
        audit(session, user, "adjudication.resolve." + data.resolution, case.id)
        session.commit()
        return dispute_record(case, passage)

    @app.get("/v1/passages/{passage_id}/adjudications")
    def list_adjudications(passage_id: str, user: dict = Depends(role("curator", "admin")),
                           session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        cases = session.scalars(select(AnnotationDispute).where(
            AnnotationDispute.passage_id == passage_id).order_by(AnnotationDispute.created_at, AnnotationDispute.id)).all()
        return {"passage_id": passage_id, "unresolved": unresolved_count(session, passage_id),
                "cases": [dispute_record(c, passage) for c in cases]}

    @app.get("/v1/documents/{document_id}/duplicates")
    def possible_duplicates(document_id: str, user: dict = Depends(role("curator", "admin")),
                            session: Session = Depends(db)):
        document = get_doc(session, document_id, user)
        candidates = []
        for other in session.scalars(select(Document).where(Document.id != document_id).order_by(Document.id)):
            if not visible(other, user):
                continue
            match = signals(document, other)
            if not match:
                continue
            a, b = ordered_pair(document.id, other.id)
            decision = session.scalar(select(BibliographicDecision).where(
                BibliographicDecision.left_document_id == a, BibliographicDecision.right_document_id == b))
            candidates.append({"document_id": other.id, "title": other.title, "doi": other.doi,
                "year": other.year, "signals": match,
                "verdict": decision.verdict if decision else "unreviewed",
                "reviewed_by": decision.reviewed_by if decision else None})
        return {"document_id": document_id, "candidates": candidates, "automatically_merged": False}

    @app.post("/v1/bibliography/review")
    def review_bibliography(data: BibliographyVerdict,
                            user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        try:
            a, b = ordered_pair(data.left_document_id, data.right_document_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        left, right = get_doc(session, a, user), get_doc(session, b, user)
        if not signals(left, right):
            raise HTTPException(422, "Sem sinais bibliográficos; encaminhe para revisão manual externa")
        # Revisão independente de quem cadastrou os PDFs; evita autocertificação.
        uploaders = session.scalars(select(AuditEvent.actor).where(
            AuditEvent.action == "document.upload", AuditEvent.resource.in_((a, b)))).all()
        if user["actor"] in uploaders:
            raise HTTPException(403, "A decisão bibliográfica exige revisor distinto do responsável pela ingestão")
        existing = session.scalar(select(BibliographicDecision).where(
            BibliographicDecision.left_document_id == a,
            BibliographicDecision.right_document_id == b))
        if existing:
            raise HTTPException(409, "Par já decidido; preservamos a decisão e a trilha de auditoria")
        decision = BibliographicDecision(left_document_id=a, right_document_id=b,
            verdict=data.verdict, reviewed_by=user["actor"], reason=data.reason.strip())
        session.add(decision)
        audit(session, user, "bibliography.review." + data.verdict, a + ":" + b)
        commit_unique(session, "Par bibliográfico decidido em outra transação")
        return {"id": decision.id, "left_document_id": a, "right_document_id": b,
                "verdict": decision.verdict, "automatically_merged": False,
                "training_group_id": publication_clusters(session)[a]}

    @app.get("/v1/passages/{passage_id}/relation-candidates")
    def relation_candidates(passage_id: str, user: dict = Depends(role("curator", "admin")),
                            session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        candidates = candidate_pairs(session, passage.id)
        return {"passage_id": passage.id, "status": "candidate_unverified" if candidates else "no_candidates",
                "warning": "Proximidade de entidades não prova nenhuma relação científica",
                "candidates": candidates}

    @app.post("/v1/relations", status_code=201)
    def propose_relation(data: RelationProposal, user: dict = Depends(role("curator", "admin")),
                         session: Session = Depends(db)):
        passage = session.get(Passage, data.passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        key, approved = certified_anchors(session, passage.id)
        if key is None:
            raise HTTPException(409, "Certifique integralmente o trecho antes de propor relações")
        source, target = approved.get(data.subject_annotation_id), approved.get(data.object_annotation_id)
        if source is None or target is None or source.id == target.id:
            raise HTTPException(422, "Os dois argumentos devem ser anotações aprovadas e distintas do mesmo trecho")
        if (source.label, target.label) != RELATION_SIGNATURES[data.relation_type]:
            raise HTTPException(422, "Assinatura dos rótulos incompatível com a relação")
        begin, end = min(source.start, target.start), max(source.end, target.end)
        if not (data.evidence_start <= begin and end <= data.evidence_end <= len(passage.text)):
            raise HTTPException(422, "A evidência literal deve conter integralmente os dois argumentos")
        if data.evidence_end - data.evidence_start > MAX_EVIDENCE_CHARS:
            raise HTTPException(422, "Evidência muito extensa; selecione um trecho menor")
        existing = session.scalar(select(ScientificRelation).where(
            ScientificRelation.passage_id == passage.id,
            ScientificRelation.certification_key == key,
            ScientificRelation.subject_annotation_id == source.id,
            ScientificRelation.object_annotation_id == target.id,
            ScientificRelation.relation_type == data.relation_type,
        ))
        if existing:
            raise HTTPException(409, "Esta relação já foi proposta para esta certificação")
        rel = ScientificRelation(passage_id=passage.id, subject_annotation_id=source.id,
            object_annotation_id=target.id, relation_type=data.relation_type,
            evidence_start=data.evidence_start, evidence_end=data.evidence_end,
            certification_key=key, proposed_by=user["actor"])
        session.add(rel)
        session.flush()
        audit(session, user, "relation.propose", rel.id)
        session.commit()
        return {"id": rel.id, "status": rel.status, "relation_type": rel.relation_type,
                "evidence_quote": passage.text[rel.evidence_start:rel.evidence_end]}

    @app.post("/v1/relations/{relation_id}/review")
    def review_relation(relation_id: str, data: RelationReviewDecision,
                        user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        rel = session.get(ScientificRelation, relation_id)
        if rel is None:
            raise HTTPException(404, "Relação não encontrada")
        passage = session.get(Passage, rel.passage_id)
        get_doc(session, passage.document_id, user)
        if rel.status != "pending":
            raise HTTPException(409, "Relação já revisada")
        if rel.proposed_by == user["actor"]:
            raise HTTPException(403, "Uma segunda identidade deve revisar a relação")
        key, approved = certified_anchors(session, rel.passage_id)
        if key != rel.certification_key or not key or (approved.get(rel.subject_annotation_id) is None
            or approved.get(rel.object_annotation_id) is None):
            raise HTTPException(409, "Certificação alterada; relação obsoleta")
        if data.explicit_no_relation and data.decision != "rejected":
            raise HTTPException(422, "Negativo explícito exige rejeição da relação")
        rel.status = data.decision
        rel.reviewed_by = user["actor"]
        rel.reviewed_at = now()
        if data.explicit_no_relation:
            session.add(RelationNegativeAssessment(relation_id=rel.id,
                assessed_by=user["actor"], reason="relation_absent"))
        audit(session, user, "relation.review." + data.decision, rel.id)
        session.commit()
        return {"id": rel.id, "status": rel.status, "verified": is_current(session, rel)}

    @app.get("/v1/passages/{passage_id}/relations")
    def relations_for_passage(passage_id: str, user: dict = Depends(role("curator", "admin")),
                              session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        rows = session.scalars(select(ScientificRelation).where(
            ScientificRelation.passage_id == passage_id).order_by(ScientificRelation.created_at, ScientificRelation.id)).all()
        current, _ = certified_anchors(session, passage_id)
        return {"passage_id": passage_id, "relations": [{
            "id": r.id, "relation_type": r.relation_type, "status": r.status,
            "certification_current": current == r.certification_key,
            "verified": is_current(session, r),
            "subject_annotation_id": r.subject_annotation_id,
            "object_annotation_id": r.object_annotation_id,
            "evidence_quote": passage.text[r.evidence_start:r.evidence_end],
            "proposed_by": r.proposed_by, "reviewed_by": r.reviewed_by,
        } for r in rows]}

    @app.get("/v1/relations/search")
    def search_relations(q: str | None = Query(default=None, min_length=2, max_length=300),
                         relation_type: str | None = None,
                         limit: int = Query(default=20, ge=1, le=50),
                         user: dict = Depends(role(*ROLES)), session: Session = Depends(db)):
        if relation_type is not None and relation_type not in RELATION_SIGNATURES:
            raise HTTPException(422, "Tipo de relação não reconhecido")
        rows = session.execute(select(ScientificRelation, Passage, Document)
            .join(Passage, Passage.id == ScientificRelation.passage_id)
            .join(Document, Document.id == Passage.document_id)
            .where(ScientificRelation.status == "approved")
            .order_by(Document.sha256, Passage.page, ScientificRelation.id)).all()
        verified = []
        for rel, passage, doc in rows:
            if (relation_type and rel.relation_type != relation_type) or not visible(doc, user):
                continue
            if not is_current(session, rel):
                continue
            _, anchors = certified_anchors(session, passage.id)
            record = relation_record(rel, passage, doc, anchors)
            verified.append(record)
        if q is not None:
            contexts = [f"{r['subject']['quote']} {r['object']['quote']} {r['evidence']['quote']}" for r in verified]
            weights = bm25(q, contexts)
            ordered = sorted(((weights[i], r) for i, r in enumerate(verified) if weights[i] > 0),
                             key=lambda t: (-t[0], t[1]["id"]))
            verified = [r for _, r in ordered]
        return {"status": "ok" if verified else "insufficient_evidence", "scientifically_validated": False,
                "scope": "human_reviewed_relations_not_externally_validated",
                "results": verified[:limit]}

    def run_relation_training(run_id: str, epochs: int, seed: int):
        with job_lock, session_factory() as session:
            job = session.get(RelationModelRun, run_id)
            job.status = "running"
            session.commit()
            try:
                metrics = train_relations(session, root / "relations" / run_id, epochs=epochs, seed=seed)
                job.status = "completed"
                job.artifact_dir = str(root / "relations" / run_id)
                job.metrics_json = json.dumps(metrics)
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)[:1000]
            session.commit()

    @app.post("/v1/relation-models/train", status_code=202)
    def start_relation_training(data: TrainRequest, user: dict = Depends(role("admin",)),
                                session: Session = Depends(db)):
        job = RelationModelRun(status="queued")
        session.add(job)
        session.flush()
        audit(session, user, "relation_model.training.start", job.id)
        session.commit()
        executor.submit(run_relation_training, job.id, data.epochs, data.seed)
        return {"id": job.id, "status": job.status}

    @app.get("/v1/relation-models/runs/{run_id}")
    def relation_model_status(run_id: str, user: dict = Depends(role("curator", "admin")),
                              session: Session = Depends(db)):
        run = session.get(RelationModelRun, run_id)
        if run is None:
            raise HTTPException(404, "Treinamento de relações não encontrado")
        return {"id": run.id, "status": run.status,
                "metrics": json.loads(run.metrics_json) if run.metrics_json else None,
                "error": run.error}

    @app.get("/v1/relation-models/runs/{run_id}/evaluation")
    def relation_evaluation(run_id: str, user: dict = Depends(role("admin",)),
                            session: Session = Depends(db)):
        """Avaliação interna: ids/hashes de documentos e erros são restritos ao admin."""
        run = session.get(RelationModelRun, run_id)
        if run is None:
            raise HTTPException(404, "Execução não encontrada")
        if run.status != "completed" or not run.artifact_dir:
            raise HTTPException(409, "A avaliação ainda não foi concluída")
        folder = Path(run.artifact_dir).resolve()
        if not folder.is_relative_to((root / "relations").resolve()) or folder.name != run.id:
            raise HTTPException(409, "Diretório de avaliação não autorizado")
        try:
            report = verified_report(folder)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            raise HTTPException(409, "Integridade da avaliação inválida") from None
        return report

    @app.get("/v1/passages/{passage_id}/relation-predictions")
    def relation_predictions(passage_id: str, run_id: str | None = None,
                             user: dict = Depends(role("curator", "admin")),
                             session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        document = get_doc(session, passage.document_id, user)
        if run_id is not None:
            run = session.get(RelationModelRun, run_id)
        else:
            run = session.scalar(select(RelationModelRun).where(RelationModelRun.status == "completed")
                .order_by(RelationModelRun.created_at.desc(), RelationModelRun.id.desc()))
        base = {"passage_id": passage_id, "document_id": document.id, "page": passage.page,
                "doi": document.doi, "scientifically_validated": False}
        if run is None or run.status != "completed" or not run.artifact_dir:
            return {**base, "status": "insufficient_evidence", "reason": "no_trained_relation_model",
                    "candidates": []}
        raw = candidate_pairs(session, passage_id)
        if not raw:
            return {**base, "status": "insufficient_evidence", "reason": "no_certified_pairs",
                    "model_run_id": run.id, "candidates": []}
        _, anchors = certified_anchors(session, passage_id)
        if len(raw) > 100:
            raise HTTPException(422, "Muitas relações candidatas: refine a anotação; nenhuma sugestão foi omitida")
        enriched = []
        for candidate in raw:
            subj = anchors[candidate["subject_annotation_id"]]
            dest = anchors[candidate["object_annotation_id"]]
            enriched.append({**candidate, "subject": (subj.start, subj.end), "object": (dest.start, dest.end)})
        try:
            ranked = predict_candidates(passage, enriched, Path(run.artifact_dir))
        except (ValueError, OSError, KeyError) as exc:
            raise HTTPException(409, "Modelo de relações indisponível ou incompatível") from exc
        return {**base, "status": "experimental_unverified", "model_run_id": run.id,
                "warning": "Escore não calibrado: nenhuma relação é aprovada automaticamente",
                "candidates": ranked}

    @app.post("/v1/agreement/tasks", status_code=201)
    def new_agreement_task(data: NewAgreementTask,
                           user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, data.passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        if not 0 <= data.start < data.end <= len(passage.text) or not passage.text[data.start:data.end].strip():
            raise HTTPException(422, "Offsets da amostra inválidos")
        a, b = data.rater_a.strip(), data.rater_b.strip()
        if a == b or not a or not b:
            raise HTTPException(422, "São necessárias duas identidades diferentes")
        # Não é permitido criar tarefas destinadas a identidades que não possuem
        # permissão de curadoria configurada para esta instância.
        configured = configured_curators
        if a not in configured or b not in configured:
            raise HTTPException(422, "Revisores precisam possuir credenciais de curadoria válidas")
        a, b = sorted((a, b))
        existing = session.scalar(select(AgreementTask).where(
            AgreementTask.passage_id == passage.id, AgreementTask.start == data.start,
            AgreementTask.end == data.end, AgreementTask.rater_a == a, AgreementTask.rater_b == b))
        if existing:
            raise HTTPException(409, "Amostra já atribuída a esta dupla; não substituímos respostas")
        task = AgreementTask(passage_id=passage.id, start=data.start, end=data.end,
                             rater_a=a, rater_b=b, created_by=user["actor"])
        session.add(task)
        flush_unique(session, "Amostra duplicada em operação concorrente")
        audit(session, user, "agreement.task.create", task.id)
        commit_unique(session, "Amostra duplicada em operação concorrente")
        return {**task_state(session, task), "quote": passage.text[task.start:task.end]}

    @app.get("/v1/agreement/tasks")
    def list_agreement_tasks(passage_id: str | None = None,
                             user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        statement = select(AgreementTask).order_by(AgreementTask.created_at.desc(), AgreementTask.id.desc()).limit(100)
        if passage_id is not None:
            passage = session.get(Passage, passage_id)
            if passage is None:
                raise HTTPException(404, "Trecho não encontrado")
            get_doc(session, passage.document_id, user)
            statement = statement.where(AgreementTask.passage_id == passage_id)
        tasks = []
        for task in session.scalars(statement):
            if user["role"] != "admin" and user["actor"] not in (task.rater_a, task.rater_b):
                continue
            passage = session.get(Passage, task.passage_id)
            document = session.get(Document, passage.document_id)
            if not visible(document, user):
                continue
            tasks.append({**task_state(session, task, reveal=True), "quote": passage.text[task.start:task.end]})
        return {"tasks": tasks, "has_more_possible": len(tasks) == 100}

    @app.get("/v1/agreement/tasks/{task_id}")
    def agreement_task_details(task_id: str, user: dict = Depends(role("curator", "admin")),
                               session: Session = Depends(db)):
        task = session.get(AgreementTask, task_id)
        if task is None:
            raise HTTPException(404, "Amostra não encontrada")
        passage = session.get(Passage, task.passage_id)
        get_doc(session, passage.document_id, user)
        if user["role"] != "admin" and user["actor"] not in (task.rater_a, task.rater_b):
            raise HTTPException(403, "Amostra destinada a outra dupla")
        # Antes dos dois votos ninguém vê a escolha do outro, nem o admin.
        return {**task_state(session, task, reveal=True), "quote": passage.text[task.start:task.end]}

    @app.post("/v1/agreement/tasks/{task_id}/vote")
    def submit_agreement_vote(task_id: str, data: AgreementLabel,
                              user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        task = session.get(AgreementTask, task_id)
        if task is None:
            raise HTTPException(404, "Amostra não encontrada")
        passage = session.get(Passage, task.passage_id)
        get_doc(session, passage.document_id, user)
        if user["actor"] not in (task.rater_a, task.rater_b):
            raise HTTPException(403, "Apenas revisores designados podem votar")
        if user["actor"] in votes_for(session, task_id):
            raise HTTPException(409, "Voto já registrado; observações independentes são imutáveis")
        if session.get(AgreementResolution, task_id):
            raise HTTPException(409, "Amostra já adjudicada")
        session.add(AgreementVote(task_id=task.id, actor=user["actor"], label=data.label))
        audit(session, user, "agreement.vote", task.id)
        commit_unique(session, "Voto já registrado em operação concorrente")
        # resposta não inclui votos — mantenha cegamento até conclusão.
        return task_state(session, task)

    @app.post("/v1/agreement/tasks/{task_id}/adjudicate")
    def adjudicate_agreement(task_id: str, data: AgreementAdjudication,
                             user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        task = session.get(AgreementTask, task_id)
        if task is None:
            raise HTTPException(404, "Amostra não encontrada")
        passage = session.get(Passage, task.passage_id)
        get_doc(session, passage.document_id, user)
        if user["actor"] in (task.rater_a, task.rater_b):
            raise HTTPException(403, "Decisão final exige terceira identidade")
        state = task_state(session, task, reveal=True)
        if not state["complete"] or not state["needs_adjudication"]:
            raise HTTPException(409, "Somente discordâncias completas e pendentes são adjudicáveis")
        session.add(AgreementResolution(task_id=task.id, actor=user["actor"],
                                        label=data.label, reason=data.reason.strip()))
        audit(session, user, "agreement.adjudicate", task.id)
        commit_unique(session, "Decisão já registrada por outra transação")
        return task_state(session, task, reveal=True)

    @app.get("/v1/agreement/metrics")
    def agreement_metrics(user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        return agreement_report(session)

    @app.get("/v1/datasets/gold/export")
    def export_gold(user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        try:
            snapshot = gold_snapshot(session)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        # Payload restrito a admin; o conteúdo pode conter material institucional.
        audit(session, user, "gold.export", snapshot["sha256"])
        session.commit()
        return snapshot

    @app.post("/v1/datasets/gold/snapshots", status_code=201)
    def save_gold_snapshot(user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        try:
            record, created = create_snapshot(session, actor=user["actor"])
            audit(session, user, "gold.snapshot.create" if created else "gold.snapshot.reuse", record.sha256)
            commit_unique(session, "Snapshot já criado em outra transação; consulte pela hash")
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"sha256": record.sha256, "created": created, "schema": "bia.scientific_gold.v1"}

    @app.get("/v1/datasets/gold/snapshots")
    def list_gold_snapshots(user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        rows = session.scalars(select(DatasetSnapshot).order_by(DatasetSnapshot.created_at.desc()).limit(100)).all()
        return {"snapshots": [{"sha256": r.sha256, "created_by": r.created_by,
                               "created_at": r.created_at.isoformat()} for r in rows]}

    @app.get("/v1/datasets/gold/snapshots/{digest}")
    def get_gold_snapshot(digest: str, user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        row = session.get(DatasetSnapshot, digest)
        if row is None:
            raise HTTPException(404, "Snapshot não encontrado")
        try:
            data = current_snapshot(session, row)
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        audit(session, user, "gold.snapshot.read", digest)
        session.commit()
        return data

    def institute_member(institute: str, user: dict):
        if institute not in INSTITUTES:
            raise HTTPException(404, "Instituto não reconhecido")
        membership = user.get("institutes", [])
        if not isinstance(membership, (tuple, list)) or institute not in membership:
            raise HTTPException(403, "Afiliação não habilitada para este instituto")

    @app.get("/v1/institutes/{institute}/protocol")
    def institutional_protocol(institute: str, user: dict = Depends(role("curator", "admin"))):
        institute_member(institute, user)
        return {"institute": institute, "blind_to_gold": True,
                "review_unit": "predeclared_same_span_entity_label", "labels": sorted(AGREEMENT_LABELS),
                "one_assessment_per_task": True, "scientifically_validated": False}

    @app.post("/v1/institutes/{institute}/assessments", status_code=201)
    def institutional_assessment(institute: str, data: InstituteEvaluation,
                                 user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        institute_member(institute, user)
        task = session.get(AgreementTask, data.task_id)
        if task is None:
            raise HTTPException(404, "Amostra não encontrada")
        passage = session.get(Passage, task.passage_id)
        get_doc(session, passage.document_id, user)
        state = task_state(session, task, reveal=True)
        key = certification_key(session.get(PassageReview, task.passage_id))
        if not key or not passage_agreement_ready(session, task.passage_id) or not state["complete"] or state["final_label"] is None:
            raise HTTPException(409, "Gold não está concluído para esta amostra")
        resolution = session.get(AgreementResolution, task.id)
        if user["actor"] in (task.rater_a, task.rater_b) or (resolution and user["actor"] == resolution.actor):
            raise HTTPException(403, "Avaliador precisa ser independente do gold")
        if session.scalar(select(InstituteAssessment).where(
            InstituteAssessment.task_id == task.id, InstituteAssessment.institute == institute,
            InstituteAssessment.certification_key == key)):
            raise HTTPException(409, "Instituto já avaliou esta amostra")
        record = InstituteAssessment(task_id=task.id, institute=institute, certification_key=key,
                                     actor=user["actor"], label=data.label)
        session.add(record)
        session.flush()
        audit(session, user, "institute.assessment.create", record.id)
        commit_unique(session, "Avaliação institucional concorrente já registrada")
        return {"id": record.id, "institute": institute, "task_id": task.id, "status": "recorded_blind"}

    @app.get("/v1/institutes/assessments/report")
    def institutional_report(user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        return institute_report(session)

    def run_training(run_id: str, epochs: int, seed: int):
        with job_lock, session_factory() as session:
            row = session.get(TrainingRun, run_id)
            row.status = "running"
            session.commit()
            try:
                metrics = train(session, root / run_id, epochs=epochs, seed=seed)
                row.status = "completed"
                row.artifact_dir = str(root / run_id)
                row.metrics_json = json.dumps(metrics)
            except Exception as exc:
                row.status = "failed"
                row.error = str(exc)[:1000]
            session.commit()

    @app.post("/v1/models/train", status_code=202)
    def start_training(data: TrainRequest, user: dict = Depends(role("admin",)), session: Session = Depends(db)):
        job = TrainingRun(status="queued")
        session.add(job)
        session.flush()
        audit(session, user, "model.training.start", job.id)
        session.commit()
        executor.submit(run_training, job.id, data.epochs, data.seed)
        return {"id": job.id, "status": job.status}

    @app.get("/v1/models/runs/{run_id}")
    def training_status(run_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        run = session.get(TrainingRun, run_id)
        if run is None:
            raise HTTPException(404, "Treinamento não encontrado")
        return {"id": run.id, "status": run.status, "metrics": json.loads(run.metrics_json) if run.metrics_json else None,
                "error": run.error}

    @app.get("/v1/extract/{passage_id}")
    def extract_entities(passage_id: str, user: dict = Depends(role(*ROLES)), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        document = get_doc(session, passage.document_id, user)
        run = session.scalar(select(TrainingRun).where(TrainingRun.status == "completed")
                             .order_by(TrainingRun.created_at.desc(), TrainingRun.id.desc()))
        base = {"document_id": document.id, "doi": document.doi, "passage_id": passage.id, "page": passage.page}
        if run is None or not run.artifact_dir:
            return {**base, "status": "insufficient_evidence", "reason": "no_trained_model", "entities": []}
        predictions = extract(passage.text, Path(run.artifact_dir))
        return {**base, "status": "experimental", "model_run_id": run.id, **predictions}

    return app


app = make_app()
