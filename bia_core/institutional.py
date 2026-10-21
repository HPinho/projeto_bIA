"""BIA-02.6: snapshots imutáveis e protocolo de avaliação tripartite.

Afiliação dos avaliadores deve ser declarada na configuração de autenticação.
Resultados NÃO representam validação institucional de um modelo sem amostragem e
revisão científicas externas.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json

from sqlalchemy import select

from .agreement import AGREEMENT_LABELS, canonical_bytes, gold_snapshot, task_state, validate_gold_snapshot, passage_agreement_ready
from .db import AgreementTask, DatasetSnapshot, Document, InstituteAssessment, Passage, PassageReview
from .relations import certification_key

INSTITUTES = ("inovaligno", "polissacarideos", "circularidade")
REPORT_SCHEMA = "bia.institute_assessments.v1"


def current_snapshot(session, row: DatasetSnapshot) -> dict:
    """Bloqueia reexposição de texto histórico após revogação de direitos."""
    try:
        stored = json.loads(row.payload_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Snapshot corrompido") from exc
    if not validate_gold_snapshot(stored) or stored["sha256"] != row.sha256:
        raise ValueError("Hash do snapshot inválido")
    digests = {item["document_sha256"] for item in stored["rows"]}
    docs = session.scalars(select(Document).where(Document.sha256.in_(digests))).all()
    if len(docs) != len(digests) or any(not d.training_allowed or d.status != "ready" for d in docs):
        raise PermissionError("Snapshot não pode ser recuperado: direito de treinamento revogado")
    return stored


def create_snapshot(session, *, actor: str) -> tuple[DatasetSnapshot, bool]:
    snapshot = gold_snapshot(session)
    if not snapshot["rows"] or not validate_gold_snapshot(snapshot):
        raise ValueError("Snapshot vazio ou inválido; certifique documentos autorizados")
    prior = session.get(DatasetSnapshot, snapshot["sha256"])
    if prior is not None:
        current_snapshot(session, prior)
        return prior, False
    row = DatasetSnapshot(sha256=snapshot["sha256"], payload_json=canonical_bytes(snapshot).decode("utf-8"),
                          created_by=actor)
    session.add(row)
    session.flush()
    return row, True


def institute_report(session) -> dict:
    """Métricas descritivas somente em tarefas gold concluídas, nunca para treinar."""
    tasks = {t.id: t for t in session.scalars(select(AgreementTask)).all()}
    validated = {}
    for tid, task in tasks.items():
        passage = session.get(Passage, task.passage_id)
        review = session.get(PassageReview, task.passage_id)
        document = session.get(Document, passage.document_id) if passage else None
        if not review or review.status != "certified" or document is None or not document.training_allowed:
            continue
        if not passage_agreement_ready(session, task.passage_id):
            continue
        state = task_state(session, task, reveal=True)
        if state["complete"] and state["final_label"] is not None:
            validated[tid] = state["final_label"]
    rows = session.scalars(select(InstituteAssessment).order_by(InstituteAssessment.id)).all()
    summary = []
    for institute in INSTITUTES:
        selected = [r for r in rows if r.institute == institute and r.task_id in validated
                    and r.certification_key == certification_key(
                        session.get(PassageReview, tasks[r.task_id].passage_id))]
        confusion = Counter((validated[r.task_id], r.label) for r in selected)
        total = len(selected)
        matches = sum(count for (gold, prediction), count in confusion.items() if gold == prediction)
        recall = []
        for label in AGREEMENT_LABELS:
            positives = sum(count for (expected, _), count in confusion.items() if expected == label)
            if positives:
                recall.append({"label": label, "support": positives,
                               "recall": confusion.get((label, label), 0) / positives})
        summary.append({"institute": institute, "eligible_tasks": len(validated),
                        "assessed_tasks": total, "not_assessed": len(validated) - total,
                        "exact_agreement": matches / total if total else None,
                        "recall_by_label": recall,
                        "confusion": [{"gold": gold, "assessment": pred, "count": count}
                                      for (gold, pred), count in sorted(confusion.items())]})
    value = {"schema": REPORT_SCHEMA, "scientifically_validated": False,
             "protocol": "one_blinded_assessment_per_institute_and_gold_task",
             "limitations": "Amostragem e independência devem ser certificadas pelos INCTs; acordo com gold não é desempenho do modelo.",
             "institutes": summary}
    return {**value, "sha256": sha256(canonical_bytes(value)).hexdigest()}
