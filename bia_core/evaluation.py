"""BIA-02.3 — relatórios determinísticos de avaliação, sem inferência científica indevida.

O conjunto de teste é isolado do treinamento. Relatórios exportam apenas
identificadores e hashes (nunca conteúdo integral de documentos ou tokens).
Uma métrica sem exemplos de suporte é ausente (null), nunca zero inventado.
"""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path


def sha256_bytes(content: bytes) -> str:
    return sha256(content).hexdigest()


def _division(top: float, bottom: int) -> float | None:
    return top / bottom if bottom else None


def summarise_predictions(records: list[dict], relation_types: tuple[str, ...]) -> dict:
    """Métricas binárias por tipo e agregado; positivos = relações presentes.

    `precision` e `recall` ficam null se denominador for zero; F1 só é
    definido se ambos têm denominadores, evitando falsa certeza.
    """
    known = set(relation_types)
    counts = {name: {"tp": 0, "tn": 0, "fp": 0, "fn": 0} for name in relation_types}
    for row in records:
        label, prediction, relation = row["label"], row["prediction"], row["relation_type"]
        if relation not in known or label not in (0, 1) or prediction not in (0, 1):
            raise ValueError("Predição ou rótulo de avaliação inválido")
        key = ("t" if label == prediction else "f") + ("p" if prediction == 1 else "n")
        counts[relation][key] += 1

    def rates(c: dict) -> dict:
        tp, tn, fp, fn = (c[k] for k in ("tp", "tn", "fp", "fn"))
        precision = _division(tp, tp + fp)
        recall = _division(tp, tp + fn)
        f1 = (2 * tp / (2 * tp + fp + fn)) if precision is not None and recall is not None and (2 * tp + fp + fn) else None
        return {"examples": tp + tn + fp + fn, "positives": tp + fn, "negatives": tn + fp,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1}

    by_type = {name: rates(c) for name, c in counts.items()}
    aggregate = {k: sum(c[k] for c in counts.values()) for k in ("tp", "tn", "fp", "fn")}
    macro = [v["f1"] for v in by_type.values() if v["f1"] is not None]
    return {"overall": {**rates(aggregate), "macro_f1_defined_types": _division(sum(macro), len(macro)),
                         "types_with_defined_f1": len(macro)},
            "by_type": by_type}


def build_evaluation_report(records: list[dict], *, relation_types: tuple[str, ...],
                            assignments: dict[str, str], dataset_sha256: str) -> dict:
    """Gerar análise de erros e avaliação documental de validação/teste.

    NÃO retorna texto, citações completas ou identificadores de pessoas.
    """
    if len(dataset_sha256) != 64 or any(x not in "0123456789abcdef" for x in dataset_sha256):
        raise ValueError("Identificador de dataset inválido")
    for row in records:
        if assignments.get(row["document_sha256"]) != row["split"]:
            raise ValueError("Avaliação inconsistente com divisão dos documentos")
    exported: dict[str, dict] = {}
    for split in ("validation", "test"):
        relevant = [r for r in records if r["split"] == ("val" if split == "validation" else "test")]
        relevant.sort(key=lambda r: (r["document_sha256"], r["relation_id"]))
        metrics = summarise_predictions(relevant, relation_types)
        errors = [{"relation_id": r["relation_id"], "document_sha256": r["document_sha256"],
                   "passage_sha256": r["passage_sha256"], "relation_type": r["relation_type"],
                   "expected": r["label"], "predicted": r["prediction"],
                   "score_uncalibrated": r["score_uncalibrated"],
                   "kind": "false_positive" if r["prediction"] else "false_negative"}
                  for r in relevant if r["prediction"] != r["label"]]
        per_document = defaultdict(list)
        for r in relevant:
            per_document[r["document_sha256"]].append(r)
        metrics.update({
            "document_count": len(per_document),
            "by_document": {key: summarise_predictions(rows, relation_types)["overall"]
                            for key, rows in sorted(per_document.items())},
            "errors": errors,
            "evaluable": bool(relevant),
        })
        exported[split] = metrics
    train_docs = {doc for doc, split in assignments.items() if split == "train"}
    val_docs = {doc for doc, split in assignments.items() if split == "val"}
    test_docs = {doc for doc, split in assignments.items() if split == "test"}
    if train_docs & val_docs or train_docs & test_docs or val_docs & test_docs:
        raise ValueError("Vazamento: publicação aparece em mais de um conjunto")
    return {
        "schema_version": 1, "dataset_sha256": dataset_sha256,
        "scientifically_validated": False,
        "score_calibrated": False,
        "protocol": "training_only_on_train; validation_for_development; test_held_out_post_training",
        "documents": {"train": len(train_docs), "validation": len(val_docs), "test": len(test_docs)},
        "validation": exported["validation"], "test": exported["test"],
        "limitations": ["Conjuntos pequenos podem impedir estimativas confiáveis",
                        "Casos sintéticos não estabelecem desempenho científico real",
                        "O conjunto de teste não deve orientar escolha de hiperparâmetros"],
    }


def verified_report(artifact_dir: Path) -> dict:
    """Somente leitura de relatórios produzidos e com hash no manifest."""
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("architecture") != "bia-relation-char-transformer" or manifest.get("evaluation_schema_version") != 1:
        raise ValueError("Avaliação de arquitetura incompatível ou indisponível")
    report_path = artifact_dir / "evaluation_report.json"
    snapshot_path = artifact_dir / "dataset_snapshot.json"
    for path, expected in ((report_path, manifest.get("evaluation_sha256")),
                           (snapshot_path, manifest.get("dataset_sha256"))):
        if not isinstance(expected, str) or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError("Integridade do relatório ou dataset comprometida")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset_sha256") != manifest["dataset_sha256"] or report.get("schema_version") != 1:
        raise ValueError("Relatório e conjunto de dados incompatíveis")
    return report
