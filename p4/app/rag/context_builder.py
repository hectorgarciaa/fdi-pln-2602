"""Construcción del contexto que se entrega al LLM."""

from __future__ import annotations

from dataclasses import dataclass, replace

from p4.app.models import RagSource
from p4.app.utils import normalize_whitespace


@dataclass(slots=True)
class RagContext:
    """Contexto serializado para el LLM junto con las fuentes seleccionadas."""

    prompt_context: str
    sources: list[RagSource]
    total_chars: int


@dataclass(slots=True)
class ContextBuilder:
    """Selecciona fuentes y limita el contexto total para RAG."""

    max_sources: int
    max_context_chars: int
    max_source_chars: int

    def build(self, sources: list[RagSource]) -> RagContext:
        """Asigna identificadores de fuente y empaqueta el contexto textual."""

        if self.max_sources <= 0 or self.max_context_chars <= 0:
            return RagContext(prompt_context="", sources=[], total_chars=0)

        selected: list[RagSource] = []
        blocks: list[str] = []
        total_chars = 0

        for source in sources[: self.max_sources]:
            source_id = f"F{len(selected) + 1}"
            prepared_source = replace(source, source_id=source_id)
            header = self._format_header(prepared_source)
            remaining_budget = self.max_context_chars - total_chars - len(header) - 2
            if remaining_budget <= 0:
                break

            max_chars = min(self.max_source_chars, remaining_budget)
            text = self._truncate_text(prepared_source.text, max_chars)
            if not text:
                continue

            block = f"{header}\n{text}"
            if total_chars + len(block) > self.max_context_chars and selected:
                break

            selected.append(prepared_source)
            blocks.append(block)
            total_chars += len(block) + 2

        return RagContext(
            prompt_context="\n\n".join(blocks),
            sources=selected,
            total_chars=max(total_chars - 2, 0),
        )

    def _format_header(self, source: RagSource) -> str:
        paragraph_span = source.metadata.get("paragraph_span", "-")
        retrieval_modes = (
            ", ".join(source.explanation.get("retrieval_modes", [])) or "-"
        )
        return "\n".join(
            [
                f"[{source.source_id}]",
                f"Parte: {source.part}",
                f"Capítulo: {source.title}",
                f"Chunk: {source.chunk_id}",
                f"Párrafos: {paragraph_span}",
                f"Recuperación: {retrieval_modes}",
                "Texto original:",
            ]
        )

    def _truncate_text(self, text: str, max_chars: int) -> str:
        clean_text = normalize_whitespace(text)
        if max_chars <= 0:
            return ""
        if len(clean_text) <= max_chars:
            return clean_text
        if max_chars <= 3:
            return clean_text[:max_chars]
        return clean_text[: max_chars - 3].rstrip() + "..."
