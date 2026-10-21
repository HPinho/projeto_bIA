"""BM25 implementado na bIA, sem embeddings externos."""
from __future__ import annotations

from collections import Counter
import math
import re
import unicodedata


TOKEN = re.compile(r"[^\W_]+(?:[.,/-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    norm = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN.findall(norm)


def bm25(query: str, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    if not corpus:
        return []
    q = set(tokenize(query))
    counts = [Counter(tokenize(text)) for text in corpus]
    lengths = [sum(counts_row.values()) for counts_row in counts]
    avg_len = sum(lengths) / len(lengths) or 1.0
    dfs = Counter(word for row in counts for word in row)
    scores = []
    for row, length in zip(counts, lengths):
        score = 0.0
        for term in q:
            tf = row.get(term, 0)
            if not tf:
                continue
            idf = math.log(1 + (len(corpus) - dfs[term] + 0.5) / (dfs[term] + 0.5))
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
        scores.append(score)
    return scores
