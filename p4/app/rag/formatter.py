"""Formateo de respuestas y fuentes del modo RAG."""

from __future__ import annotations

from p4.app.models import RagResponse, RagSource


def referenced_sources(response: RagResponse) -> list[RagSource]:
    """Devuelve las fuentes citadas por la respuesta, manteniendo el orden."""

    reference_set = set(response.references)
    return [source for source in response.sources if source.source_id in reference_set]


def format_rag_answer_markdown(response: RagResponse) -> str:
    """Construye un resumen markdown para TUI/CLI."""

    lines = [
        "## Respuesta RAG",
        "",
        response.answer or "No se generó una respuesta.",
        "",
        "### Fuentes recuperadas",
    ]

    for source in response.sources:
        used_flag = (
            "usada" if source.source_id in set(response.references) else "recuperada"
        )
        paragraph_span = source.metadata.get("paragraph_span", "-")
        lines.append(
            f"- [{source.source_id}] {source.title} ({source.chunk_id}, párrafos {paragraph_span}, {used_flag})"
        )

    if not response.sources:
        lines.append("- No se recuperaron fuentes.")

    return "\n".join(lines)
