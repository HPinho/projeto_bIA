"""Treinamento local auditável, sem checkpoints ou pesos externos.

Invariante científico: somente passagens integralmente certificadas geram
classes O (background). A ausência de rótulo em um rascunho é DESCONHECIDO.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random

from sqlalchemy import select
import torch
import torch.nn.functional as F

from .curation import annotations_for
from .db import Document, Passage, PassageReview
from .model import BiaTagger, ENTITY_LABELS, LABELS, ModelConfig
from .tokenizer import CharTokenizer


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def source_dataset(session) -> list[dict]:
    """Certificação dupla, direitos explícitos; zero entidades é negativo válido."""
    records = session.execute(
        select(Document, Passage, PassageReview)
        .join(Passage, Document.id == Passage.document_id)
        .join(PassageReview, PassageReview.passage_id == Passage.id)
        .where(Document.training_allowed.is_(True), PassageReview.status == "certified")
        .order_by(Document.sha256, Passage.page, Passage.id)
    ).all()
    rows = []
    for doc, passage, review in records:
        annotations = annotations_for(session, passage.id)
        if any(a.status == "pending" for a in annotations):
            raise ValueError("Trecho certificado contém anotações pendentes")
        approved = sorted(
            ({"start": a.start, "end": a.end, "label": a.label} for a in annotations if a.status == "approved"),
            key=lambda a: (a["start"], a["end"], a["label"]),
        )
        rows.append({
            "document_id": doc.id, "sha256": doc.sha256, "passage_id": passage.id,
            "page": passage.page, "text": passage.text, "annotations": approved,
            "submitted_by": review.submitted_by, "certified_by": review.certified_by,
        })
    return rows


def document_splits(rows: list[dict]) -> dict[str, str]:
    """Holdouts determinísticos agrupados por documento, não por trecho."""
    docs = sorted({item["sha256"] for item in rows})
    rng = random.Random(101)
    rng.shuffle(docs)
    if len(docs) < 3:
        return {key: "train" for key in docs}
    n_test = max(1, len(docs) // 10)
    n_val = max(1, len(docs) // 10)
    return {key: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, key in enumerate(docs)}


def build_samples(rows: list[dict], tokenizer: CharTokenizer, max_length: int):
    """Particionar sem cortar entidades BIO; falhar explicitamente se impossível."""
    if max_length < 1:
        raise ValueError("max_length deve ser positivo")
    samples = []
    inside = {LABELS.index("I-" + label) for label in ENTITY_LABELS}
    for row in rows:
        text = row["text"]
        labels = [0] * len(text)
        for ann in row["annotations"]:
            start, end, label = ann["start"], ann["end"], ann["label"]
            if label not in ENTITY_LABELS or not 0 <= start < end <= len(text):
                raise ValueError("Anotação certificada possui rótulo ou offsets inválidos")
            if end - start > max_length:
                raise ValueError("Entidade certificada maior que max_length; ajuste o modelo")
            for pos in range(start, end):
                if labels[pos] != 0:
                    raise ValueError("Anotações sobrepostas devem ser resolvidas antes do treino")
                labels[pos] = LABELS.index(("B-" if pos == start else "I-") + label)
        start = 0
        while start < len(text):
            end = min(start + max_length, len(text))
            if end < len(text) and labels[end] in inside:
                while end > start and labels[end] in inside:
                    end -= 1
                # end agora aponta para o B da entidade que cruza a borda.
                if end == start:
                    raise ValueError("Entidade não cabe na janela sem perda de evidência")
            if end <= start:
                raise ValueError("Falha na segmentação BIO")
            ids = tokenizer.encode(text[start:end])
            samples.append((row["sha256"], ids, labels[start:end]))
            start = end
    return samples


def _spans(labels: list[int]) -> set[tuple[int, int, str]]:
    """Entidades completas por trecho (offset relativo à janela)."""
    entities = set()
    active = None
    for offset, idx in enumerate(labels + [0]):
        tag = LABELS[idx]
        kind = tag[2:] if tag != "O" else None
        if active and (not tag.startswith("I-") or kind != active[2]):
            entities.add(active)
            active = None
        if kind and active is None:
            active = (offset, offset + 1, kind)
        elif active and tag.startswith("I-") and kind == active[2]:
            active = (active[0], offset + 1, active[2])
    return entities


def train(session, out_dir: Path, *, epochs: int = 2, seed: int = 42, config_override: dict | None = None) -> dict:
    if not (1 <= epochs <= 100):
        raise ValueError("epochs fora da faixa permitida")
    rows = source_dataset(session)
    if not rows:
        raise ValueError("Não há trechos certificados em documentos autorizados para treinamento")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    split_map = document_splits(rows)
    # Nenhum vocabulário é ajustado com dados de validação/teste.
    tokenizer = CharTokenizer.fit([item["text"] for item in rows if split_map[item["sha256"]] == "train"])
    config = ModelConfig(vocab_size=len(tokenizer.vocab), **(config_override or {}))
    config.validate()
    samples = build_samples(rows, tokenizer, config.max_length)
    train_samples = [(ids, labels) for key, ids, labels in samples if split_map[key] == "train"]
    val_samples = [(ids, labels) for key, ids, labels in samples if split_map[key] == "val"]
    if not train_samples:
        raise ValueError("Divisão de dados não gerou amostras de treino")
    model = BiaTagger(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    losses = []
    for _ in range(epochs):
        for ids, target in train_samples:
            x = torch.tensor([ids], dtype=torch.long)
            y = torch.tensor([target], dtype=torch.long)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, len(LABELS)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    model.eval()
    tp = fp = fn = 0
    with torch.inference_mode():
        for ids, target in val_samples:
            predicted = model(torch.tensor([ids], dtype=torch.long)).argmax(-1)[0].tolist()
            expected_set = _spans(target)
            predicted_set = _spans(predicted)
            tp += len(expected_set & predicted_set)
            fp += len(predicted_set - expected_set)
            fn += len(expected_set - predicted_set)
    metrics = {
        "train_documents": sum(s == "train" for s in split_map.values()),
        "validation_documents": sum(s == "val" for s in split_map.values()),
        "test_documents": sum(s == "test" for s in split_map.values()),
        "certified_passages": len(rows),
        "train_examples": len(train_samples),
        "validation_examples": len(val_samples),
        "test_examples": sum(split_map[key] == "test" for key, _, _ in samples),
        "train_last_loss": losses[-1], "scientifically_validated": False,
        "validation_entity_precision": tp / (tp + fp) if val_samples and tp + fp else None,
        "validation_entity_recall": tp / (tp + fn) if val_samples and tp + fn else None,
        "validation_entity_f1": 2 * tp / (2 * tp + fp + fn) if val_samples and 2 * tp + fp + fn else None,
        "note": "F1 por span e rótulo BIO exato; avaliação científica externa ainda necessária.",
    }
    # Criar o diretório apenas após a avaliação do dataset, sem sobrescrever checkpoints.
    out_dir.mkdir(parents=True, exist_ok=False)
    weights = out_dir / "weights.pt"
    torch.save(model.state_dict(), weights)
    tokenizer.save(out_dir / "tokenizer.json")
    snapshot = [{
        "document_sha256": item["sha256"], "passage_id": item["passage_id"], "page": item["page"],
        "passage_sha256": _hash(item["text"].encode()), "split": split_map[item["sha256"]],
        "annotations": item["annotations"], "submitted_by": item["submitted_by"],
        "certified_by": item["certified_by"],
    } for item in rows]
    snapshot_path = out_dir / "dataset_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "architecture": "bia-char-transformer", "seed": seed, "epochs": epochs,
        "config": asdict(config), "labels": LABELS, "metrics": metrics,
        "dataset_sha256": _hash(snapshot_path.read_bytes()),
        "weights_sha256": _hash(weights.read_bytes()),
        "tokenizer_sha256": _hash((out_dir / "tokenizer.json").read_bytes()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def extract(text: str, artifact_dir: Path) -> dict:
    """Predições experimentais, sempre apoiadas em trechos literais."""
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    weights_path = artifact_dir / "weights.pt"
    tokenizer_path = artifact_dir / "tokenizer.json"
    if _hash(weights_path.read_bytes()) != manifest["weights_sha256"] or _hash(tokenizer_path.read_bytes()) != manifest["tokenizer_sha256"]:
        raise ValueError("Checkpoint inválido: hash não confere")
    tokenizer = CharTokenizer.load(tokenizer_path)
    model = BiaTagger(ModelConfig(**manifest["config"]))
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    predicted = []
    with torch.inference_mode():
        for start in range(0, len(text), model.config.max_length):
            ids = tokenizer.encode(text[start:start + model.config.max_length])
            if ids:
                predicted.extend(model(torch.tensor([ids], dtype=torch.long)).argmax(-1)[0].tolist())
    entities = []
    current = None
    for index, label_id in enumerate(predicted):
        label = LABELS[label_id]
        if label.startswith("B-") or (label.startswith("I-") and (current is None or current["label"] != label[2:])):
            if current is not None:
                entities.append(current)
            current = {"label": label[2:], "start": index, "end": index + 1}
        elif label.startswith("I-") and current is not None and current["label"] == label[2:]:
            current["end"] = index + 1
        else:
            if current is not None:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    for entity in entities:
        entity["quote"] = text[entity["start"]:entity["end"]]
    return {"model_sha256": manifest["weights_sha256"], "scientifically_validated": False, "entities": entities}
