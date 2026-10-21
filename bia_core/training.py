"""Treino CPU local: seed, splits por documento, checkpoints sem pesos externos."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random

from sqlalchemy import select
import torch
import torch.nn.functional as F

from .db import Annotation, Document, Passage
from .model import BiaTagger, ENTITY_LABELS, LABELS, ModelConfig
from .tokenizer import CharTokenizer


def _hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def source_dataset(session):
    """Somente documentos com permissão explícita de treino e rótulos aprovados."""
    rows = session.execute(
        select(Document, Passage, Annotation)
        .join(Passage, Document.id == Passage.document_id)
        .join(Annotation, Annotation.passage_id == Passage.id)
        .where(Document.training_allowed.is_(True), Annotation.status == "approved")
    ).all()
    grouped: dict[str, dict] = {}
    for doc, passage, ann in rows:
        entry = grouped.setdefault(passage.id, {
            "document_id": doc.id, "sha256": doc.sha256, "text": passage.text, "annotations": [],
        })
        entry["annotations"].append({"start": ann.start, "end": ann.end, "label": ann.label})
    return list(grouped.values())


def document_splits(rows: list[dict]) -> dict[str, str]:
    """Split estável por hash de documento; nunca dividir páginas do mesmo artigo."""
    docs = sorted(set(item["sha256"] for item in rows))
    rng = random.Random(101)
    rng.shuffle(docs)
    if len(docs) < 3:
        return {key: "train" for key in docs}
    n_test = max(1, len(docs) // 10)
    n_val = max(1, len(docs) // 10)
    return {key: ("test" if i < n_test else "val" if i < n_test + n_val else "train") for i, key in enumerate(docs)}


def build_samples(rows: list[dict], tokenizer: CharTokenizer, max_length: int):
    samples = []
    for row in rows:
        text = row["text"]
        labels = [0] * len(text)
        for ann in row["annotations"]:
            b = LABELS.index("B-" + ann["label"])
            i = LABELS.index("I-" + ann["label"])
            for pos in range(ann["start"], ann["end"]):
                if labels[pos] != 0:
                    raise ValueError("Anotações sobrepostas devem ser resolvidas antes do treino")
                labels[pos] = b if pos == ann["start"] else i
        for start in range(0, len(text), max_length):
            end = min(start + max_length, len(text))
            if start and labels[start] and labels[start] != 0 and labels[start] not in [LABELS.index("B-" + key) for key in ENTITY_LABELS]:
                # Uma entidade cruzando fronteira não deve virar evidência artificial.
                continue
            ids = tokenizer.encode(text[start:end])
            if ids:
                samples.append((row["sha256"], ids, labels[start:end]))
    return samples


def train(session, out_dir: Path, *, epochs: int = 2, seed: int = 42, config_override: dict | None = None) -> dict:
    if not (1 <= epochs <= 100):
        raise ValueError("epochs fora da faixa permitida")
    rows = source_dataset(session)
    if not rows:
        raise ValueError("Não há anotações aprovadas em documentos autorizados para treinamento")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    tokenizer = CharTokenizer.fit([item["text"] for item in rows])
    config = ModelConfig(vocab_size=len(tokenizer.vocab), **(config_override or {}))
    config.validate()
    split_map = document_splits(rows)
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
            predictions = model(torch.tensor([ids], dtype=torch.long)).argmax(-1)[0].tolist()
            for pred, true in zip(predictions, target):
                if pred != 0 and pred == true:
                    tp += 1
                elif pred != 0:
                    fp += 1
                if true != 0 and pred != true:
                    fn += 1
    metrics = {
        "train_examples": len(train_samples), "validation_examples": len(val_samples),
        "test_examples": sum(split_map[key] == "test" for key, _, _ in samples),
        "train_last_loss": losses[-1], "scientifically_validated": False,
        "validation_micro_precision": tp / (tp + fp) if val_samples and tp + fp else None,
        "validation_micro_recall": tp / (tp + fn) if val_samples and tp + fn else None,
        "validation_micro_f1": 2 * tp / (2 * tp + fp + fn) if val_samples and 2 * tp + fp + fn else None,
        "note": "Métricas de tokens BIO, não equivalem à precisão científica por entidade.",
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    weights = out_dir / "weights.pt"
    torch.save(model.state_dict(), weights)
    tokenizer.save(out_dir / "tokenizer.json")
    snapshot = [{"document_sha256": item["sha256"], "passage_sha256": _hash(item["text"].encode()), "split": split_map[item["sha256"]]} for item in rows]
    (out_dir / "dataset_snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"architecture": "bia-char-transformer", "seed": seed, "epochs": epochs, "config": asdict(config),
                "labels": LABELS, "metrics": metrics, "weights_sha256": _hash(weights.read_bytes()),
                "tokenizer_sha256": _hash((out_dir / "tokenizer.json").read_bytes())}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def extract(text: str, artifact_dir: Path) -> dict:
    """Predições experimentais; toda entidade retorna o trecho original como evidência."""
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
            if not ids:
                continue
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
