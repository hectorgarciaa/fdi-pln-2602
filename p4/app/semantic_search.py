"""Búsqueda semántica con vectores de spaCy `es_core_news_lg`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from p4.app.errors import SemanticModelError
from p4.app.models import Chunk, SearchResult, TextAnalysis
from p4.app.utils import extract_fragment


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class SpacyVectorEmbedder:
    """Genera embeddings semánticos con el vector de documento de spaCy."""

    REQUIRED_MODEL = "es_core_news_lg"

    def __init__(self, nlp, model_name: str, batch_size: int = 32) -> None:
        self.model = model_name
        self._nlp = nlp
        self.batch_size = batch_size
        if model_name != self.REQUIRED_MODEL:
            raise SemanticModelError(
                "La búsqueda semántica de esta práctica debe usar spaCy con el modelo "
                f"{self.REQUIRED_MODEL!r}. Ajusta `spacy_model` y reconstruye los embeddings."
            )
        if getattr(self._nlp.vocab, "vectors_length", 0) <= 0:
            raise SemanticModelError(
                "El modelo de spaCy cargado no incluye vectores semánticos. "
                f"Instala y usa {self.REQUIRED_MODEL!r}."
            )

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Genera embeddings para una lista de textos."""

        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[np.ndarray] = []
        for doc in self._nlp.pipe(clean_texts, batch_size=self.batch_size):
            vector = np.asarray(doc.vector, dtype=np.float32)
            if vector.size == 0:
                raise SemanticModelError(
                    "spaCy no devolvió vectores de documento válidos. "
                    f"Comprueba que {self.REQUIRED_MODEL!r} está instalado correctamente."
                )
            vectors.append(vector)

        embeddings = np.vstack(vectors).astype(np.float32)
        if embeddings.ndim != 2:
            raise SemanticModelError(
                "spaCy devolvió embeddings con un formato inesperado."
            )
        return embeddings


@dataclass(slots=True)
class SemanticSearchEngine:
    """Motor de similitud coseno sobre embeddings persistidos."""

    chunk_ids: list[str]
    embeddings: np.ndarray
    model: str
    index_version: int
    title_weight: float
    body_weight: float

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: SpacyVectorEmbedder,
        *,
        index_version: int,
        title_weight: float,
        body_weight: float,
    ) -> "SemanticSearchEngine":
        if not chunks:
            raise SemanticModelError(
                "No se pueden construir embeddings semánticos sin chunks."
            )

        title_vectors = embedder.embed_texts([chunk.title for chunk in chunks])
        body_vectors = embedder.embed_texts([chunk.text for chunk in chunks])
        embeddings = _blend_vectors(
            title_vectors=title_vectors,
            body_vectors=body_vectors,
            title_weight=title_weight,
            body_weight=body_weight,
        )
        return cls(
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            model=embedder.model,
            index_version=index_version,
            title_weight=title_weight,
            body_weight=body_weight,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "chunk_ids": self.chunk_ids,
            "model": self.model,
            "index_version": self.index_version,
            "title_weight": self.title_weight,
            "body_weight": self.body_weight,
            "dimensions": int(self.embeddings.shape[1]) if self.embeddings.size else 0,
        }

    @classmethod
    def from_manifest(
        cls, manifest: dict[str, Any], embeddings: np.ndarray
    ) -> "SemanticSearchEngine":
        return cls(
            chunk_ids=list(manifest["chunk_ids"]),
            embeddings=_normalize_rows(embeddings.astype(np.float32)),
            model=str(manifest["model"]),
            index_version=int(manifest.get("index_version", 1)),
            title_weight=float(manifest.get("title_weight", 0.0)),
            body_weight=float(manifest.get("body_weight", 1.0)),
        )

    def search(
        self,
        query: str,
        analysis: TextAnalysis,
        chunks: list[Chunk],
        embedder: SpacyVectorEmbedder,
        top_k: int,
        *,
        original_query_weight: float,
        normalized_query_weight: float,
        lexical_bonus_weight: float,
        rerank_pool_size: int,
    ) -> list[SearchResult]:
        """Ejecuta una búsqueda semántica por similitud coseno."""

        query_vector = _build_query_vector(
            query=query,
            analysis=analysis,
            embedder=embedder,
            original_query_weight=original_query_weight,
            normalized_query_weight=normalized_query_weight,
        )
        if query_vector.size == 0:
            return []

        scores = self.embeddings @ query_vector
        ranked_indices = np.argsort(scores)[::-1]
        rerank_limit = min(max(rerank_pool_size, top_k), len(ranked_indices))
        lexical_terms = set(analysis.surface_tokens + analysis.lemma_tokens)

        results: list[SearchResult] = []
        rescored: list[tuple[float, float, float, int]] = []
        for document_index in ranked_indices[:rerank_limit]:
            semantic_score = float(scores[document_index])
            if semantic_score <= 0:
                continue
            lexical_bonus = _lexical_bonus(
                lexical_terms=lexical_terms, chunk=chunks[int(document_index)]
            )
            final_score = semantic_score + (lexical_bonus_weight * lexical_bonus)
            rescored.append(
                (final_score, semantic_score, lexical_bonus, int(document_index))
            )

        rescored.sort(key=lambda item: item[0], reverse=True)
        for final_score, semantic_score, lexical_bonus, document_index in rescored:
            if final_score <= 0:
                continue
            chunk = chunks[document_index]
            results.append(
                SearchResult(
                    rank=len(results) + 1,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    part=chunk.part,
                    title=chunk.title,
                    score=round(final_score, 6),
                    fragment=extract_fragment(
                        chunk.text, [query, *chunk.lemma_tokens[:10]]
                    ),
                    text=chunk.text,
                    metadata={
                        "chunk_word_count": chunk.word_count,
                        "paragraph_span": f"{chunk.start_paragraph}-{chunk.end_paragraph}",
                        "paragraph_count": chunk.metadata.get("paragraph_count"),
                    },
                    explanation={
                        "mode": "semantic",
                        "semantic_score": round(semantic_score, 6),
                        "lexical_bonus": round(lexical_bonus, 6),
                        "lexical_bonus_weight": lexical_bonus_weight,
                        "query_surface_terms": analysis.surface_tokens,
                        "query_lemma_terms": analysis.lemma_tokens,
                        "embedding_model": self.model,
                        "vector_backend": "spaCy document vectors",
                        "index_version": self.index_version,
                        "title_weight": self.title_weight,
                        "body_weight": self.body_weight,
                        "query": query,
                    },
                )
            )
            if len(results) >= top_k:
                break
        return results


def _blend_vectors(
    *,
    title_vectors: np.ndarray,
    body_vectors: np.ndarray,
    title_weight: float,
    body_weight: float,
) -> np.ndarray:
    if title_vectors.size == 0 or body_vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    total_weight = title_weight + body_weight
    if total_weight <= 0:
        raise SemanticModelError(
            "Los pesos semánticos de título y cuerpo deben sumar más que cero."
        )

    normalized_titles = _normalize_rows(title_vectors.astype(np.float32))
    normalized_bodies = _normalize_rows(body_vectors.astype(np.float32))
    combined = (
        (title_weight / total_weight) * normalized_titles
        + (body_weight / total_weight) * normalized_bodies
    ).astype(np.float32)
    return _normalize_rows(combined)


def _build_query_vector(
    *,
    query: str,
    analysis: TextAnalysis,
    embedder: SpacyVectorEmbedder,
    original_query_weight: float,
    normalized_query_weight: float,
) -> np.ndarray:
    components: list[tuple[float, np.ndarray]] = []
    if query.strip() and original_query_weight > 0:
        original = embedder.embed_texts([query])
        if original.size > 0:
            components.append((original_query_weight, _normalize_rows(original)[0]))

    normalized_query = analysis.normalized_text.strip()
    if normalized_query and normalized_query_weight > 0:
        normalized = embedder.embed_texts([normalized_query])
        if normalized.size > 0:
            components.append((normalized_query_weight, _normalize_rows(normalized)[0]))

    if not components:
        return np.zeros((0,), dtype=np.float32)

    total_weight = sum(weight for weight, _ in components)
    if total_weight <= 0:
        raise SemanticModelError(
            "Los pesos semánticos de consulta deben sumar más que cero."
        )

    vector = np.zeros_like(components[0][1], dtype=np.float32)
    for weight, component in components:
        vector += (weight / total_weight) * component
    return _normalize_rows(np.asarray([vector], dtype=np.float32))[0]


def _lexical_bonus(*, lexical_terms: set[str], chunk: Chunk) -> float:
    if not lexical_terms:
        return 0.0
    chunk_terms = set(chunk.surface_tokens) | set(chunk.lemma_tokens)
    if not chunk_terms:
        return 0.0
    overlap = lexical_terms & chunk_terms
    return len(overlap) / len(lexical_terms)
