"""Tokenizador de caracteres treinado exclusivamente no corpus local.

Cada token não especial corresponde a UM caractere Unicode; offset é reversível.
Não possui vocabulários, pesos nem checkpoints importados.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


SPECIAL = ("<pad>", "<unk>")


class CharTokenizer:
    def __init__(self, vocab: dict[str, int]):
        if vocab.get("<pad>") != 0 or vocab.get("<unk>") != 1:
            raise ValueError("Vocabulário precisa iniciar com PAD=0, UNK=1")
        self.vocab = vocab
        self.reverse = {v: k for k, v in vocab.items()}

    @classmethod
    def fit(cls, texts: list[str], max_chars: int = 4096) -> CharTokenizer:
        counts = Counter(char for text in texts for char in text)
        chars = sorted(counts, key=lambda c: (-counts[c], ord(c)))[:max_chars]
        return cls({**{item: i for i, item in enumerate(SPECIAL)}, **{c: i + len(SPECIAL) for i, c in enumerate(chars)}})

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(char, 1) for char in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.reverse[i] for i in ids if i >= 2)

    def save(self, path: Path):
        path.write_text(json.dumps(self.vocab, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CharTokenizer:
        return cls(json.loads(path.read_text(encoding="utf-8")))
