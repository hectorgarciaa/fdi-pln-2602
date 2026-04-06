"""Generación RAG apoyada en Ollama."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from p4.app.errors import ConfigurationError, RagGenerationError
from p4.app.models import RagResponse
from p4.app.rag.context_builder import RagContext
from p4.app.rag.prompts import build_system_prompt, build_user_prompt
from p4.app.utils import normalize_whitespace


REFERENCE_RE = re.compile(r"\[(F\d+)\]")
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.DOTALL)


@dataclass(slots=True)
class OllamaRagGenerator:
    """Coordina prompting, llamada a Ollama y validación de referencias."""

    host: str
    model: str
    temperature: float
    num_predict: int
    timeout_seconds: float

    def generate(self, query: str, context: RagContext) -> RagResponse:
        """Genera una respuesta grounded en las fuentes recuperadas."""

        if not self.model.strip():
            raise ConfigurationError(
                "RAG requiere un modelo de generación configurado en `rag.generation_model` "
                "o mediante `QUIJOTE_RAG__GENERATION_MODEL`."
            )
        if not context.sources:
            raise RagGenerationError(
                "No se puede generar una respuesta RAG sin fuentes recuperadas."
            )

        try:
            from ollama import Client, RequestError, ResponseError
        except ModuleNotFoundError as exc:
            raise ConfigurationError(
                "La dependencia `ollama` no está disponible en el entorno de p4. "
                "Ejecuta `uv sync --project p4`."
            ) from exc

        client = Client(host=self.host, timeout=self.timeout_seconds)
        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": build_user_prompt(query, context)},
                ],
                stream=False,
                format="json",
                options={
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            )
        except ConnectionError as exc:
            raise ConfigurationError(
                "No se pudo conectar con Ollama en "
                f"{self.host}. Asegúrate de que el servicio esté activo "
                "(`ollama serve`) y vuelve a intentarlo."
            ) from exc
        except RequestError as exc:
            raise ConfigurationError(
                "No se pudo conectar con Ollama en "
                f"{self.host}. Asegúrate de que el servicio esté activo "
                "(`ollama serve`) y vuelve a intentarlo."
            ) from exc
        except ResponseError as exc:
            if getattr(exc, "status_code", -1) == 404:
                raise ConfigurationError(
                    "El modelo configurado para RAG no está disponible en Ollama: "
                    f"{self.model!r}. Descárgalo con `ollama pull {self.model}` "
                    "o ajusta `rag.generation_model`."
                ) from exc
            raise ConfigurationError(
                f"Ollama devolvió un error al generar la respuesta RAG: {exc}"
            ) from exc

        raw_content = normalize_whitespace(response.message.content or "")
        payload = self._parse_payload(raw_content)
        answer = normalize_whitespace(str(payload.get("answer", "")))
        insufficient_evidence = bool(payload.get("insufficient_evidence", False))
        if not answer:
            raise RagGenerationError("Ollama devolvió una respuesta RAG vacía.")

        valid_references = {source.source_id for source in context.sources}
        cited_in_answer = self._extract_references(answer)
        if unknown := cited_in_answer.difference(valid_references):
            raise RagGenerationError(
                "El modelo devolvió referencias inexistentes en el contexto: "
                + ", ".join(sorted(unknown))
            )

        used_sources = self._normalize_used_sources(
            payload.get("used_sources", []), valid_references
        )
        if not cited_in_answer:
            if used_sources:
                answer = (
                    answer.rstrip()
                    + "\n\nFuentes: "
                    + " ".join(f"[{source_id}]" for source_id in used_sources)
                )
                cited_in_answer = set(used_sources)
            elif insufficient_evidence:
                fallback_refs = [source.source_id for source in context.sources[:2]]
                if fallback_refs:
                    answer = (
                        answer.rstrip()
                        + "\n\nFuentes consultadas: "
                        + " ".join(f"[{source_id}]" for source_id in fallback_refs)
                    )
                    used_sources = fallback_refs
                    cited_in_answer = set(fallback_refs)
            else:
                raise RagGenerationError(
                    "La respuesta RAG no incluyó referencias válidas al contexto recuperado."
                )

        if not used_sources:
            used_sources = [
                source.source_id
                for source in context.sources
                if source.source_id in cited_in_answer
            ]

        ordered_references = [
            source.source_id
            for source in context.sources
            if source.source_id in set(used_sources) | cited_in_answer
        ]

        logger.info(
            "Respuesta RAG generada con {} fuentes y modelo {}",
            len(ordered_references),
            self.model,
        )

        return RagResponse(
            query=query,
            answer=answer,
            sources=context.sources,
            references=ordered_references,
            model=self.model,
            context=context.prompt_context,
            raw_response=raw_content,
            metadata={
                "mode": "rag",
                "insufficient_evidence": insufficient_evidence,
                "used_source_count": len(ordered_references),
                "available_source_count": len(context.sources),
                "context_chars": context.total_chars,
                "ollama_host": self.host,
            },
        )

    def _parse_payload(self, raw_content: str) -> dict[str, Any]:
        content = raw_content.strip()
        content = CODE_FENCE_RE.sub("", content).strip()
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RagGenerationError(
                "Ollama no devolvió un JSON válido para la respuesta RAG."
            ) from exc
        if not isinstance(payload, dict):
            raise RagGenerationError(
                "El formato devuelto por Ollama para RAG no es un objeto JSON."
            )
        return payload

    def _extract_references(self, answer: str) -> set[str]:
        return {match.group(1) for match in REFERENCE_RE.finditer(answer)}

    def _normalize_used_sources(
        self, raw_sources: Any, valid_references: set[str]
    ) -> list[str]:
        if not isinstance(raw_sources, list):
            return []
        normalized: list[str] = []
        for item in raw_sources:
            if not isinstance(item, str):
                continue
            source_id = item.strip().upper()
            if source_id in valid_references and source_id not in normalized:
                normalized.append(source_id)
        return normalized
