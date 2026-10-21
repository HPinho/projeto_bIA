"""Extração de PDF sem OCR ou IA externa. Offsets são do texto extraído, nunca do PDF visual."""
from __future__ import annotations

from io import BytesIO
from pypdf import PdfReader


class InvalidPDF(ValueError):
    pass


def pages_from_pdf(contents: bytes, max_pages: int = 2000) -> list[tuple[int, str]]:
    if not contents.startswith(b"%PDF-"):
        raise InvalidPDF("O arquivo não possui cabeçalho PDF válido")
    try:
        reader = PdfReader(BytesIO(contents), strict=True)
        if reader.is_encrypted:
            raise InvalidPDF("PDF protegido por senha requer revisão")
        if len(reader.pages) > max_pages:
            raise InvalidPDF("PDF excede o limite de páginas")
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    except InvalidPDF:
        raise
    except Exception as exc:
        raise InvalidPDF("PDF inválido ou sem suporte; encaminhe à revisão") from exc
