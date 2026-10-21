"""BIA-02.5: observações cegas, concordância por dupla fixa e exportação auditável.

A amostra é definida *antes* dos dois votos, não a partir de conflitos.
Um rótulo NONE se refere somente ao intervalo indicado, nunca à página inteira.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from sqlalchemy import select

from .db import AgreementTask, AgreementVote, AgreementResolution, AnnotationDispute
from .model import ENTITY_LABELS
from .training import source_dataset, document_splits

AGREEMENT_LABELS = tuple(sorted(ENTITY_LABELS)) + ("NONE",)
SCHEMA = "bia.scientific_gold.v1"


def votes_for(session, task_id: str) -> dict[str, str]:
    return {vote.actor: vote.label for vote in session.scalars(
        select(AgreementVote).where(AgreementVote.task_id == task_id)).all()}


def task_state(session, task: AgreementTask, *, reveal: bool = False) -> dict:
    opinions = votes_for(session, task.id)
    complete = task.rater_a in opinions and task.rater_b in opinions
    resolution = session.get(AgreementResolution, task.id)
    result = {
        "id": task.id, "passage_id": task.passage_id, "start": task.start, "end": task.end,
        "raters": [task.rater_a, task.rater_b], "votes_received": len(opinions),
        "complete": complete,
        "needs_adjudication": complete and opinions[task.rater_a] != opinions[task.rater_b] and resolution is None,
        "adjudicated": resolution is not None,
    }
    if reveal and complete:
        result["votes"] = {"a": opinions[task.rater_a], "b": opinions[task.rater_b]}
        result["final_label"] = resolution.label if resolution else (
            opinions[task.rater_a] if opinions[task.rater_a] == opinions[task.rater_b] else None)
        if resolution:
            result["adjudication"] = {"actor": resolution.actor, "reason": resolution.reason}
    return result


def passage_agreement_ready(session, passage_id: str) -> bool:
    """Exige decisão final e consistência literal com entidades certificadas."""
    from .curation import annotations_for
    approved = [a for a in annotations_for(session, passage_id) if a.status == "approved"]
    tasks = session.scalars(select(AgreementTask).where(AgreementTask.passage_id == passage_id)).all()
    for task in tasks:
        state = task_state(session, task, reveal=True)
        if not state["complete"] or state["needs_adjudication"]:
            return False
        final = state["final_label"]
        overlapping = [a for a in approved if a.start < task.end and task.start < a.end]
        if final == "NONE":
            if overlapping:
                return False
        elif len(overlapping) != 1 or (overlapping[0].start, overlapping[0].end, overlapping[0].label) != (task.start, task.end, final):
            return False
    return True


def _metrics(pairs: list[tuple[str, str]], total_tasks: int) -> dict:
    count = len(pairs)
    matching = sum(a == b for a, b in pairs)
    marginal_a, marginal_b = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = matching / count if count else None
    expected = (sum(marginal_a[c] * marginal_b[c] for c in AGREEMENT_LABELS) / count**2) if count else None
    kappa = ((observed - expected) / (1 - expected)) if count and expected < 1 else None
    confusion = [{"a": a, "b": b, "count": n}
                 for (a, b), n in sorted(Counter(pairs).items())]
    return {"tasks": total_tasks, "complete": count, "incomplete": total_tasks - count,
            "agreement_rate": observed, "expected_agreement": expected, "cohen_kappa": kappa,
            "kappa_defined": kappa is not None,
            "confusion": confusion, "raters_a": dict(sorted(marginal_a.items())),
            "raters_b": dict(sorted(marginal_b.items()))}


def agreement_report(session) -> dict:
    """Kappa somente por par FIXO de pesquisadores; nunca mistura pares distintos."""
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    counts: Counter = Counter()
    pending_adjudications = 0
    tasks = session.scalars(select(AgreementTask).order_by(AgreementTask.id)).all()
    for task in tasks:
        key = (task.rater_a, task.rater_b)
        counts[key] += 1
        state = task_state(session, task, reveal=True)
        if state["complete"]:
            pairs[key].append((state["votes"]["a"], state["votes"]["b"]))
            pending_adjudications += int(state["needs_adjudication"])
    return {
        "unit": "predeclared_same_span_entity_label",
        "sampling_warning": "Interprete resultados apenas para os intervalos pré-definidos; amostras não aleatórias não representam todo o acervo.",
        "scientifically_validated": False,
        "tasks": len(tasks), "pending_adjudications": pending_adjudications,
        "by_fixed_rater_pair": [
            {"rater_a": a, "rater_b": b, **_metrics(pairs[(a, b)], counts[(a, b)])}
            for a, b in sorted(counts)
        ],
    }


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def gold_snapshot(session) -> dict:
    """Exportação interna controlada: textos SOMENTE com autorização de treinamento."""
    rows = source_dataset(session)
    for row in rows:
        if not passage_agreement_ready(session, row["passage_id"]):
            raise ValueError("Amostras de concordância pendentes ou divergentes bloqueiam o gold")
    splits = document_splits(rows)
    output = []
    for row in rows:
        # Identidades dos revisores não são exportadas junto com o dataset.
        output.append({
            "document_sha256": row["sha256"], "publication_group": row["publication_group"],
            "passage_id": row["passage_id"], "page": row["page"],
            "text": row["text"], "annotations": row["annotations"],
            "split": splits[row["sha256"]],
        })
    output.sort(key=lambda x: (x["document_sha256"], x["page"], x["passage_id"]))
    eligible_passage_ids = {r["passage_id"] for r in rows}
    adjudicated_count = sum(1 for dispute in session.scalars(select(AnnotationDispute)).all()
                            if dispute.passage_id in eligible_passage_ids and dispute.status == "resolved")
    manifest = {
        "schema": SCHEMA,
        "intended_use": "restricted_local_training_and_scientific_review",
        "redistribution_allowed": False,
        "scientifically_validated": False,
        "passages": len(output),
        "publications": len({r["publication_group"] for r in output}),
        "adjudicated_disputes": adjudicated_count,
        "rows": output,
    }
    return {**manifest, "sha256": sha256(canonical_bytes(manifest)).hexdigest()}


def validate_gold_snapshot(snapshot: dict) -> bool:
    if not isinstance(snapshot, dict) or snapshot.get("schema") != SCHEMA:
        return False
    try:
        return sha256(canonical_bytes({k: v for k, v in snapshot.items() if k != "sha256"})).hexdigest() == snapshot.get("sha256")
    except (TypeError, ValueError):
        return False
