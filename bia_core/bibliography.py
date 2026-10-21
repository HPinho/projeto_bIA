"""Detecção conservadora de duplicatas bibliográficas; sem mesclar fontes.

DOI idêntico é sinal forte, mas não elimina direitos, arquivos ou divergências.
Título idêntico + mesmo ano é apenas sugestão, nunca identidade automática.
Decisões positivas alimentam o agrupamento conservador dos holdouts.
"""
from __future__ import annotations

from hashlib import sha256
import re
import unicodedata
from sqlalchemy import select
from sqlalchemy.orm import Session
from .db import BibliographicDecision, Document

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)


def normalize_doi(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):].strip()
            break
    return value if len(value) <= 255 and DOI_RE.fullmatch(value) else None


def title_key(title: str) -> str:
    plain = unicodedata.normalize("NFKD", title.casefold())
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch))
    return " ".join(re.findall(r"\w+", plain))


def signals(a: Document, b: Document) -> list[str]:
    if a.id == b.id:
        return []
    match: list[str] = []
    if a.sha256 == b.sha256:
        match.append("same_pdf_sha256")
    x, y = normalize_doi(a.doi), normalize_doi(b.doi)
    if x and x == y:
        match.append("same_normalized_doi")
    ta, tb = title_key(a.title), title_key(b.title)
    if a.year is not None and b.year is not None and a.year == b.year and len(ta) >= 20 and ta == tb:
        match.append("same_normalized_title_and_year")
    return match


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("Compare dois documentos distintos")
    return tuple(sorted((a, b)))


def publication_clusters(session: Session) -> dict[str, str]:
    """Union DOI + links humanos positivos; nunca separa um DOI em splits.

    Títulos semelhantes não unem grupos; parecer 'distinct' também não
    desagrupa DOIs idênticos, pois no treino é mais seguro manter juntos.
    """
    docs = list(session.scalars(select(Document).order_by(Document.id)))
    parent = {doc.id: doc.id for doc in docs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        pa, pb = find(a), find(b)
        parent[max(pa, pb)] = min(pa, pb)

    first_doi: dict[str, str] = {}
    for doc in docs:
        doi = normalize_doi(doc.doi)
        if doi:
            if doi in first_doi:
                union(doc.id, first_doi[doi])
            first_doi[doi] = doc.id
    for decision in session.scalars(select(BibliographicDecision).where(BibliographicDecision.verdict == "same_publication")):
        if decision.left_document_id in parent and decision.right_document_id in parent:
            union(decision.left_document_id, decision.right_document_id)
    members: dict[str, list[str]] = {}
    for doc in docs:
        members.setdefault(find(doc.id), []).append(doc.sha256)
    digests = {key: sha256("\n".join(sorted(values)).encode()).hexdigest() for key, values in members.items()}
    return {doc.id: digests[find(doc.id)] for doc in docs}
