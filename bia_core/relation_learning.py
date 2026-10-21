"""BIA-02.2 — classificador binário de relações, treinado integralmente em dados locais.

Amostras positivas/negativas vêm EXCLUSIVAMENTE de decisões explícitas de
curadoria independentes e certificações ainda vigentes. O escore não verifica
relações nem é uma probabilidade calibrada. Não há checkpoints externos.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import random

from sqlalchemy import select
from sqlalchemy.orm import Session
import torch
from torch import nn
import torch.nn.functional as F

from .db import Document, Passage, ScientificRelation, RelationNegativeAssessment
from .evaluation import build_evaluation_report
from .model import Block, ModelConfig
from .relations import RELATION_SIGNATURES, certified_anchors
from .tokenizer import CharTokenizer

RELATION_TYPES = tuple(RELATION_SIGNATURES)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


@dataclass(frozen=True)
class RelationConfig:
    vocab_size: int
    dim: int = 32
    heads: int = 4
    layers: int = 1
    max_length: int = 384

    def validate(self) -> None:
        ModelConfig(**asdict(self)).validate()
        if self.max_length > 2048:
            raise ValueError("Janela neural acima do limite permitido")


class BiaRelationClassifier(nn.Module):
    """Char Transformer próprio; pooling dos dois argumentos + contexto e predicado."""
    def __init__(self, config: RelationConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.positions = nn.Embedding(config.max_length, config.dim)
        self.blocks = nn.ModuleList([Block(ModelConfig(**asdict(config))) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.relation_embedding = nn.Embedding(len(RELATION_TYPES), config.dim)
        self.output = nn.Sequential(nn.Linear(config.dim * 4, config.dim), nn.GELU(), nn.Linear(config.dim, 2))

    def forward(self, ids: torch.Tensor, subject: tuple[int, int], target: tuple[int, int], relation_id: int):
        batch, size = ids.shape
        if batch != 1 or not 0 <= subject[0] < subject[1] <= size or not 0 <= target[0] < target[1] <= size:
            raise ValueError("Offsets da relação inválidos")
        if not 0 <= relation_id < len(RELATION_TYPES) or size > self.config.max_length:
            raise ValueError("Relação ou janela inválida")
        valid = ids != 0
        x = (self.tokens(ids) + self.positions(torch.arange(size, device=ids.device))[None]) * valid.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, valid)
        x = self.norm(x)
        context = (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        source = x[:, subject[0]:subject[1]].mean(1)
        dest = x[:, target[0]:target[1]].mean(1)
        relation = self.relation_embedding(torch.tensor([relation_id], device=ids.device))
        return self.output(torch.cat([context, source, dest, relation], dim=-1))


def publication_hash(doi: str | None) -> str | None:
    """Versões de um artigo com mesmo DOI devem permanecer no mesmo split."""
    if not doi or not doi.strip():
        return None
    normalized = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "http://dx.doi.org/", "https://dx.doi.org/", "doi:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    normalized = normalized.strip()
    return digest(normalized.encode("utf-8")) if normalized else None


def reviewed_relations(session: Session) -> list[dict]:
    """Rejeitado significa negativo explícito para AQUELE predicado/direção."""
    relations = session.execute(
        select(ScientificRelation, Passage, Document)
        .join(Passage, ScientificRelation.passage_id == Passage.id)
        .join(Document, Passage.document_id == Document.id)
        .where(Document.training_allowed.is_(True), ScientificRelation.status.in_(("approved", "rejected")))
        .order_by(Document.sha256, Passage.page, ScientificRelation.id)
    ).all()
    rows: list[dict] = []
    for rel, passage, doc in relations:
        current, anchors = certified_anchors(session, passage.id)
        if not current or current != rel.certification_key:
            continue
        source = anchors.get(rel.subject_annotation_id)
        dest = anchors.get(rel.object_annotation_id)
        if not source or not dest or source.id == dest.id:
            continue
        if rel.relation_type not in RELATION_SIGNATURES or (source.label, dest.label) != RELATION_SIGNATURES[rel.relation_type]:
            continue
        if not rel.reviewed_by or rel.reviewed_by == rel.proposed_by or rel.reviewed_at is None:
            continue
        if rel.status == "rejected":
            explicit = session.get(RelationNegativeAssessment, rel.id)
            if explicit is None or explicit.reason != "relation_absent" or explicit.assessed_by != rel.reviewed_by:
                continue
        if not (0 <= rel.evidence_start <= min(source.start, dest.start)
                and max(source.end, dest.end) <= rel.evidence_end <= len(passage.text)):
            raise ValueError("Relação curada possui evidência com offsets inconsistentes")
        rows.append({
            "relation_id": rel.id, "document_sha256": doc.sha256, "publication_sha256": publication_hash(doc.doi), "passage_id": passage.id,
            "text": passage.text, "passage_sha256": digest(passage.text.encode("utf-8")),
            "subject": (source.start, source.end), "object": (dest.start, dest.end),
            "type": rel.relation_type, "evidence": (rel.evidence_start, rel.evidence_end),
            "label": 1 if rel.status == "approved" else 0,
            "certification_key": current,
            "proposed_by": rel.proposed_by, "reviewed_by": rel.reviewed_by,
        })
    return rows


def grouped_splits(rows: list[dict], seed: int) -> dict[str, str]:
    """Não cruzar documentos com conteúdo de passagem idêntico entre splits."""
    documents = sorted({r["document_sha256"] for r in rows})
    parent = {x: x for x in documents}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        a, b = find(x), find(y)
        parent[max(a, b)] = min(a, b)

    first: dict[tuple[str, str], str] = {}
    for row in rows:
        signatures = [("passage", row["passage_sha256"])]
        if row.get("publication_sha256"):
            signatures.append(("doi", row["publication_sha256"]))
        for signature in signatures:
            if signature in first:
                union(row["document_sha256"], first[signature])
            first[signature] = row["document_sha256"]
    groups: dict[str, list[str]] = {}
    for doc in documents:
        groups.setdefault(find(doc), []).append(doc)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    assigned = {doc: "train" for doc in documents}
    if len(group_ids) >= 3:
        for kind, root in (("test", group_ids[0]), ("val", group_ids[1])):
            for doc in groups[root]:
                assigned[doc] = kind
        # Não treinar classificador de uma única classe por acidente.
        labels = {r["label"] for r in rows if assigned[r["document_sha256"]] == "train"}
        if labels != {0, 1}:
            for root in group_ids[:2]:
                for doc in groups[root]:
                    assigned[doc] = "train"
                labels = {r["label"] for r in rows if assigned[r["document_sha256"]] == "train"}
                if labels == {0, 1}:
                    break
    return assigned


def evidence_window(row: dict, max_length: int) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Corte determinístico preserva ambos os argumentos e a direção."""
    text = row["text"]
    source, dest = row["subject"], row["object"]
    first, last = min(source[0], dest[0]), max(source[1], dest[1])
    if not (0 <= first < last <= len(text)):
        raise ValueError("Offsets fora do documento")
    if last - first > max_length:
        raise ValueError("Par de entidades excede a janela: aumente max_length; não descartar evidência")
    evidence_start, evidence_end = row["evidence"]
    if not (0 <= evidence_start <= first < last <= evidence_end <= len(text)):
        raise ValueError("Evidência não contém os dois argumentos")
    desired = min(max_length, evidence_end - evidence_start)
    start = min(max(evidence_start, first - (desired - (last - first)) // 2), evidence_end - desired)
    end = start + desired
    assert start <= first and last <= end
    return text[start:end], (source[0]-start, source[1]-start), (dest[0]-start, dest[1]-start)


def _rates(tp: int, fp: int, fn: int, total: int):
    return {"precision": tp / (tp + fp) if total and tp + fp else None,
            "recall": tp / (tp + fn) if total and tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if total and 2 * tp + fp + fn else None}


def train_relations(session: Session, out_dir: Path, *, epochs: int = 2, seed: int = 42,
                    config_override: dict | None = None) -> dict:
    if not 1 <= epochs <= 100 or not 0 <= seed < 2**31:
        raise ValueError("Hiperparâmetros inválidos")
    rows = reviewed_relations(session)
    if not rows:
        raise ValueError("Não há relações revisadas e autorizadas para treinamento")
    assignments = grouped_splits(rows, seed)
    training = [r for r in rows if assignments[r["document_sha256"]] == "train"]
    if {r["label"] for r in training} != {0, 1}:
        raise ValueError("Treinamento exige exemplos positivos e negativos explicitamente revisados")
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    tok = CharTokenizer.fit([r["text"] for r in training])
    conf = RelationConfig(vocab_size=len(tok.vocab), **(config_override or {}))
    conf.validate()
    prepared = []
    for row in rows:
        content, subj, dest = evidence_window(row, conf.max_length)
        prepared.append((row, tok.encode(content), subj, dest, RELATION_TYPES.index(row["type"])))
    model = BiaRelationClassifier(conf)
    optim = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = []
    model.train()
    for _ in range(epochs):
        for row, ids, subj, dest, kind in prepared:
            if assignments[row["document_sha256"]] != "train":
                continue
            logits = model(torch.tensor([ids], dtype=torch.long), subj, dest, kind)
            loss = F.cross_entropy(logits, torch.tensor([row["label"]], dtype=torch.long))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(float(loss.detach()))
    model.eval()
    predictions: list[dict] = []
    with torch.inference_mode():
        for row, ids, subj, dest, kind in prepared:
            split = assignments[row["document_sha256"]]
            if split not in ("val", "test"):
                continue
            logits = model(torch.tensor([ids]), subj, dest, kind)
            predictions.append({
                "split": split, "relation_id": row["relation_id"],
                "document_sha256": row["document_sha256"], "passage_sha256": row["passage_sha256"],
                "relation_type": row["type"], "label": row["label"],
                "prediction": int(logits.argmax(-1).item()),
                "score_uncalibrated": float(F.softmax(logits, dim=-1)[0, 1].item()),
            })
    metrics = {
        "total_reviewed": len(rows),
        "train_examples": sum(assignments[r["document_sha256"]] == "train" for r in rows),
        "validation_examples": sum(assignments[r["document_sha256"]] == "val" for r in rows),
        "test_examples": sum(assignments[r["document_sha256"]] == "test" for r in rows),
        "train_positive": sum(r["label"] == 1 for r in training),
        "train_negative_explicit": sum(r["label"] == 0 for r in training),
        "last_train_loss": losses[-1],
        "scientifically_validated": False,
        "note": "Predição experimental binária, sem calibração; rejeições explícitas não equivalem a negativos de outros predicados.",
    }
    # Evitar sobrescrever artefatos: cada job tem diretório exclusivo.
    out_dir.mkdir(parents=True, exist_ok=False)
    weights = out_dir / "weights.pt"
    torch.save(model.state_dict(), weights)
    tok.save(out_dir / "tokenizer.json")
    snapshot = [{
        "document_sha256": r["document_sha256"], "publication_sha256": r["publication_sha256"], "passage_sha256": r["passage_sha256"],
        "passage_id": r["passage_id"], "relation_id": r["relation_id"],
        "relation_type": r["type"], "label": r["label"],
        "subject": r["subject"], "object": r["object"], "evidence": r["evidence"],
        "certification_key": r["certification_key"], "split": assignments[r["document_sha256"]],
        "proposed_by": r["proposed_by"], "reviewed_by": r["reviewed_by"],
    } for r in rows]
    snap = out_dir / "dataset_snapshot.json"
    snap.write_text(json.dumps(snapshot, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    dataset_sha = digest(snap.read_bytes())
    evaluation = build_evaluation_report(predictions, relation_types=RELATION_TYPES,
                                         assignments=assignments, dataset_sha256=dataset_sha)
    metrics["validation"] = evaluation["validation"]["overall"]
    metrics["validation_by_type"] = evaluation["validation"]["by_type"]
    metrics["test"] = evaluation["test"]["overall"]
    metrics["holdout_evaluable"] = evaluation["test"]["evaluable"]
    report_path = out_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(evaluation, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {"architecture": "bia-relation-char-transformer", "config": asdict(conf),
                "relation_types": RELATION_TYPES, "epochs": epochs, "seed": seed,
                "metrics": metrics, "dataset_sha256": dataset_sha, "evaluation_schema_version": 1,
                "evaluation_sha256": digest(report_path.read_bytes()),
                "weights_sha256": digest(weights.read_bytes()),
                "tokenizer_sha256": digest((out_dir / "tokenizer.json").read_bytes())}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def predict_candidates(passage: Passage, candidates: list[dict], artifact_dir: Path) -> list[dict]:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") != "bia-relation-char-transformer" or tuple(manifest.get("relation_types", ())) != RELATION_TYPES:
        raise ValueError("Arquitetura ou ontologia incompatível")
    weights = artifact_dir / "weights.pt"
    tok_path = artifact_dir / "tokenizer.json"
    snap = artifact_dir / "dataset_snapshot.json"
    for file, field in ((weights, "weights_sha256"), (tok_path, "tokenizer_sha256"), (snap, "dataset_sha256")):
        if digest(file.read_bytes()) != manifest[field]:
            raise ValueError("Integridade do checkpoint comprometida")
    tok = CharTokenizer.load(tok_path)
    model = BiaRelationClassifier(RelationConfig(**manifest["config"]))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval()
    results = []
    with torch.inference_mode():
        for candidate in candidates:
            row = {"text": passage.text,
                   "subject": candidate["subject"], "object": candidate["object"],
                   "evidence": (candidate["evidence_start"], candidate["evidence_end"])}
            text, source, target = evidence_window(row, model.config.max_length)
            ids = tok.encode(text)
            kind = RELATION_TYPES.index(candidate["relation_type"])
            score = F.softmax(model(torch.tensor([ids]), source, target, kind), dim=-1)[0, 1].item()
            results.append({
                "subject_annotation_id": candidate["subject_annotation_id"],
                "object_annotation_id": candidate["object_annotation_id"],
                "relation_type": candidate["relation_type"],
                "evidence_start": candidate["evidence_start"],
                "evidence_end": candidate["evidence_end"],
                "evidence_quote": passage.text[candidate["evidence_start"]:candidate["evidence_end"]],
                "model_score_uncalibrated": score,
                "status": "candidate_unverified", "scientifically_validated": False,
            })
    return sorted(results, key=lambda r: (-r["model_score_uncalibrated"], r["relation_type"], r["subject_annotation_id"], r["object_annotation_id"]))
