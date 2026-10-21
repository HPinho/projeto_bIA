"""API científica bIA. Sem rede de IA, sem downloads de modelos externos."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from sqlalchemy.orm import Session

from bia_core.db import Annotation, AuditEvent, Document, Passage, PassageReview, TrainingRun, make_session_factory, now
from bia_core.ingestion import InvalidPDF, pages_from_pdf
from bia_core.curation import annotations_for, review_state
from bia_core.model import ENTITY_LABELS
from bia_core.retrieval import bm25
from bia_core.training import extract, train

ROLES = ("viewer", "researcher", "curator", "admin")
ALLOWED_RIGHTS = ("open_access", "author_permission", "institutional_permission")


class NewAnnotation(BaseModel):
    passage_id: str
    label: Literal["MATERIAL", "PROCESS", "PROPERTY", "QUANTITY", "UNIT", "CONDITION", "APPLICATION"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]


class TrainRequest(BaseModel):
    epochs: int = Field(default=2, ge=1, le=25)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


def make_app(*, db_url: str | None = None, storage_dir: Path | None = None, keys: dict | None = None, artifact_root: Path | None = None) -> FastAPI:
    session_factory = make_session_factory(db_url)
    storage = Path(storage_dir or os.getenv("BIA_STORAGE_DIR", "./data/documents")).resolve()
    root = Path(artifact_root or os.getenv("BIA_MODEL_BASE_DIR", "./checkpoints")).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    key_config = keys if keys is not None else json.loads(os.getenv("BIA_API_KEYS", "{}"))
    if not isinstance(key_config, dict):
        raise ValueError("BIA_API_KEYS deve ser objeto JSON")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bia-train")
    job_lock = Lock()
    app = FastAPI(title="bIA Scientific Core", version="0.1.0", docs_url="/docs")
    origins = os.getenv("BIA_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o for o in origins if o], allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

    def db():
        with session_factory() as session:
            yield session

    def principal(authorization: str | None = Header(default=None)) -> dict:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Token Bearer obrigatório")
        # Constant-time comparison for configured secrets.
        import hmac
        token = authorization.removeprefix("Bearer ")
        for candidate, profile in key_config.items():
            if hmac.compare_digest(token, candidate):
                if not isinstance(profile, dict) or profile.get("role") not in ROLES or not profile.get("actor"):
                    break
                return profile
        raise HTTPException(401, "Token inválido")

    def role(*allowed):
        def require(user: dict = Depends(principal)):
            if user["role"] not in allowed:
                raise HTTPException(403, "Ação não autorizada")
            return user
        return require

    def visible(doc: Document, user: dict) -> bool:
        return doc.rights_basis != "institutional_permission" or user["role"] in ("curator", "admin")

    def audit(session, user, action, resource):
        session.add(AuditEvent(actor=user["actor"], action=action, resource=resource))

    def get_doc(session, doc_id, user):
        doc = session.get(Document, doc_id)
        if doc is None or not visible(doc, user):
            raise HTTPException(404, "Documento não encontrado")
        return doc

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "bia", "ai_provider": "local-only"}

    @app.post("/v1/documents", status_code=201)
    async def upload_document(
        file: UploadFile = File(...), title: str = Form(...), rights_basis: str = Form(...),
        training_allowed: bool = Form(False), doi: str | None = Form(None), year: int | None = Form(None),
        user: dict = Depends(role("curator", "admin")), session: Session = Depends(db),
    ):
        if rights_basis not in ALLOWED_RIGHTS:
            raise HTTPException(422, "Defina uma base de direitos permitida")
        if not 1 <= len(title.strip()) <= 500 or year is not None and not 1500 <= year <= 2100:
            raise HTTPException(422, "Título ou ano inválido")
        if doi and len(doi) > 255:
            raise HTTPException(422, "DOI excede o limite")
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
            raise HTTPException(409, {"message": "PDF já cadastrado", "document_id": prior.id})
        document_path = storage / f"{digest}.pdf"
        # No uploaded filename is ever used for filesystem paths.
        document_path.write_bytes(contents)
        document = Document(title=title.strip(), doi=doi.strip() if doi else None, year=year,
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
        return {"id": document.id, "sha256": digest, "status": document.status, "pages": len(pages)}

    @app.get("/v1/documents/{document_id}")
    def document_details(document_id: str, user: dict = Depends(role(*ROLES)), session: Session = Depends(db)):
        doc = get_doc(session, document_id, user)
        return {"id": doc.id, "title": doc.title, "doi": doc.doi, "year": doc.year,
                "rights_basis": doc.rights_basis, "training_allowed": doc.training_allowed,
                "sha256": doc.sha256, "status": doc.status}

    @app.get("/v1/documents/{document_id}/passages")
    def document_passages(document_id: str, user: dict = Depends(role(*ROLES)), session: Session = Depends(db)):
        get_doc(session, document_id, user)
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
        return {"query": q, "results": [{"score": score, "document_id": doc.id, "title": doc.title,
                 "doi": doc.doi, "page": p.page, "passage_id": p.id, "quote": p.text[:1200],
                 "rights_basis": doc.rights_basis} for score, p, doc in ranked],
                "status": "ok" if ranked else "insufficient_evidence"}

    @app.post("/v1/annotations", status_code=201)
    def annotate(data: NewAnnotation, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, data.passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
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
        items = annotations_for(session, passage_id)
        return [{"id": a.id, "label": a.label, "start": a.start, "end": a.end,
                 "quote": passage.text[a.start:a.end], "status": a.status, "actor": a.actor,
                 "reviewer": a.reviewer} for a in items]

    @app.post("/v1/annotations/{annotation_id}/review")
    def review(annotation_id: str, data: ReviewDecision, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        annotation = session.get(Annotation, annotation_id)
        if annotation is None:
            raise HTTPException(404, "Anotação não encontrada")
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

    @app.get("/v1/curation/summary")
    def curation_summary(user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        """Readiness counts, never confuse annotation volume with validated science."""
        documents = session.scalars(select(Document)).all()
        reviews = {
            row.passage_id: row.status
            for row in session.scalars(select(PassageReview)).all()
        }
        passages = session.scalars(select(Passage)).all()
        states = {"draft": 0, "submitted": 0, "certified": 0}
        eligible = 0
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
            "scientifically_validated": False,
        }

    @app.post("/v1/passages/{passage_id}/submit")
    def submit_passage(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        state = review_state(session, passage_id)
        if state["status"] != "draft":
            raise HTTPException(409, "Trecho já submetido ou certificado")
        if state["pending_annotations"]:
            raise HTTPException(409, "Revise todas as anotações pendentes antes de submeter")
        review = session.get(PassageReview, passage_id)
        if review is None:
            review = PassageReview(passage_id=passage_id, submitted_by=user["actor"])
            session.add(review)
        else:
            review.status = "submitted"
            review.submitted_by = user["actor"]
            review.submitted_at = now()
            review.certified_by = None
            review.certified_at = None
        audit(session, user, "passage.review.submit", passage_id)
        session.commit()
        return review_state(session, passage_id)

    @app.post("/v1/passages/{passage_id}/certify")
    def certify_passage(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        review = session.get(PassageReview, passage_id)
        if review is None or review.status != "submitted":
            raise HTTPException(409, "Trecho ainda não foi submetido")
        if review.submitted_by == user["actor"]:
            raise HTTPException(403, "Certificação exige um segundo pesquisador")
        if review_state(session, passage_id)["pending_annotations"]:
            raise HTTPException(409, "Ainda existem anotações pendentes")
        review.status = "certified"
        review.certified_by = user["actor"]
        review.certified_at = now()
        audit(session, user, "passage.review.certify", passage_id)
        session.commit()
        return review_state(session, passage_id)

    @app.post("/v1/passages/{passage_id}/reopen")
    def reopen_passage(passage_id: str, user: dict = Depends(role("curator", "admin")), session: Session = Depends(db)):
        passage = session.get(Passage, passage_id)
        if passage is None:
            raise HTTPException(404, "Trecho não encontrado")
        get_doc(session, passage.document_id, user)
        review = session.get(PassageReview, passage_id)
        if review is None or review.status == "draft":
            raise HTTPException(409, "Trecho já está em rascunho")
        review.status = "draft"
        review.certified_by = None
        review.certified_at = None
        audit(session, user, "passage.review.reopen", passage_id)
        session.commit()
        return review_state(session, passage_id)

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
