"""Capa de servicios que orquesta corpus, índices y búsquedas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from p4.app.chunking import build_chunks
from p4.app.classical_search import ClassicalSearchEngine
from p4.app.config import AppSettings, frontmatter_ids, load_settings
from p4.app.errors import (
    ArtifactMissingError,
    ConfigurationError,
    RagGenerationError,
    ResourceOutOfDateError,
)
from p4.app.ingestion import extract_documents
from p4.app.logging_utils import configure_logging
from p4.app.models import Chunk, Document, RagResponse, RagSource, SearchResult
from p4.app.preprocessing import SpanishTextPreprocessor
from p4.app.rag import (
    ContextBuilder,
    HybridRetriever,
    OllamaRagGenerator,
)
from p4.app.semantic_search import SemanticSearchEngine, SpacyVectorEmbedder
from p4.app.storage import (
    ensure_artifact,
    load_chunks,
    load_classical_index,
    load_semantic_embeddings,
    read_json,
    save_chunks,
    save_classical_index,
    save_semantic_embeddings,
    write_json,
)
from p4.app.utils import sha256_of_file, utc_now_iso


class QuijoteSearchService:
    """Fachada principal para construcción y consulta."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or load_settings()
        configure_logging(self.settings.logs_dir)
        self._preprocessor: SpanishTextPreprocessor | None = None

    @property
    def preprocessor(self) -> SpanishTextPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = SpanishTextPreprocessor(self.settings.spacy_model)
        return self._preprocessor

    def load_documents(self) -> list[Document]:
        """Carga el corpus original y extrae los capítulos."""

        if not self.settings.corpus_path.exists():
            raise ConfigurationError(
                f"No se encontró el corpus HTML en {self.settings.corpus_path}"
            )

        logger.info("Extrayendo capítulos desde {}", self.settings.corpus_path)
        return extract_documents(self.settings.corpus_path, frontmatter_ids())

    def build_chunks(self) -> tuple[dict[str, Any], list[Chunk]]:
        """Reconstruye el artefacto de chunks."""

        documents = self.load_documents()
        chunks = build_chunks(
            documents,
            preprocessor=self.preprocessor,
            target_words=self.settings.chunk_target_words,
            overlap_words=self.settings.chunk_overlap_words,
        )
        manifest = {
            "built_at": utc_now_iso(),
            "corpus_path": str(self.settings.corpus_path),
            "corpus_sha256": self._corpus_sha256(),
            "spacy_model": self.settings.spacy_model,
            "chunk_target_words": self.settings.chunk_target_words,
            "chunk_overlap_words": self.settings.chunk_overlap_words,
            "total_documents": len(documents),
            "total_chunks": len(chunks),
        }
        save_chunks(self.settings.chunks_path, chunks, manifest)
        logger.info("Chunks guardados en {}", self.settings.chunks_path)
        return manifest, chunks

    def load_chunks(self) -> tuple[dict[str, Any], list[Chunk]]:
        """Carga los chunks persistidos y valida que sigan vigentes."""

        manifest, chunks = load_chunks(self.settings.chunks_path)
        self._validate_chunk_manifest(manifest)
        return manifest, chunks

    def build_classical_index(self) -> dict[str, Any]:
        """Reconstruye el índice clásico a partir de los chunks actuales."""

        chunk_manifest, chunks = self._load_or_build_chunks()
        engine = ClassicalSearchEngine.build(
            chunks,
            surface_weight=self.settings.surface_weight,
            lemma_weight=self.settings.lemma_weight,
        )
        payload = {
            "manifest": {
                "built_at": utc_now_iso(),
                "chunk_manifest": chunk_manifest,
                "surface_weight": self.settings.surface_weight,
                "lemma_weight": self.settings.lemma_weight,
                "document_count": len(chunks),
            },
            "engine": engine.to_dict(),
        }
        save_classical_index(self.settings.classical_index_path, payload)
        logger.info("Índice clásico guardado en {}", self.settings.classical_index_path)
        return payload["manifest"]

    def load_classical_engine(self) -> ClassicalSearchEngine:
        """Carga el índice clásico desde disco y valida dependencias."""

        payload = load_classical_index(self.settings.classical_index_path)
        chunk_manifest, chunks = self.load_chunks()
        manifest = payload["manifest"]
        if (
            manifest["chunk_manifest"]["corpus_sha256"]
            != chunk_manifest["corpus_sha256"]
        ):
            raise ResourceOutOfDateError(
                "El índice clásico está desactualizado respecto a los chunks actuales. Ejecuta `build-classical`."
            )
        if manifest["chunk_manifest"]["total_chunks"] != len(chunks):
            raise ResourceOutOfDateError(
                "El índice clásico ya no coincide con el número de chunks actual. Ejecuta `build-classical`."
            )
        return ClassicalSearchEngine.from_dict(payload["engine"])

    def build_semantic_index(self) -> dict[str, Any]:
        """Reconstruye la matriz de embeddings semánticos."""

        chunk_manifest, chunks = self._load_or_build_chunks()
        documents = self.load_documents()
        embedder = self._semantic_embedder()
        engine = SemanticSearchEngine.build(chunks, documents, embedder)
        save_semantic_embeddings(
            self.settings.semantic_embeddings_path, engine.embeddings
        )
        manifest = {
            "built_at": utc_now_iso(),
            "chunk_manifest": chunk_manifest,
            **engine.to_manifest(),
        }
        write_json(self.settings.semantic_manifest_path, manifest)
        logger.info(
            "Embeddings semánticos guardados en {}",
            self.settings.semantic_embeddings_path,
        )
        return manifest

    def load_semantic_engine(self) -> SemanticSearchEngine:
        """Carga los embeddings semánticos persistidos y valida dependencias."""

        ensure_artifact(self.settings.semantic_manifest_path)
        manifest = read_json(self.settings.semantic_manifest_path)
        chunk_manifest, chunks = self.load_chunks()
        if (
            manifest["chunk_manifest"]["corpus_sha256"]
            != chunk_manifest["corpus_sha256"]
        ):
            raise ResourceOutOfDateError(
                "Los embeddings están desactualizados respecto al corpus. Ejecuta `build-semantic`."
            )
        if manifest["chunk_manifest"]["total_chunks"] != len(chunks):
            raise ResourceOutOfDateError(
                "Los embeddings no coinciden con los chunks actuales. Ejecuta `build-semantic`."
            )
        embeddings = load_semantic_embeddings(self.settings.semantic_embeddings_path)
        engine = SemanticSearchEngine.from_manifest(manifest, embeddings)
        if engine.chunk_ids != [chunk.chunk_id for chunk in chunks]:
            raise ResourceOutOfDateError(
                "El orden de chunks y embeddings no coincide. Ejecuta `build-semantic`."
            )
        return engine

    def build_all(self, include_semantic: bool = True) -> dict[str, Any]:
        """Reconstruye todos los recursos principales del proyecto."""

        chunk_manifest, _ = self.build_chunks()
        classical_manifest = self.build_classical_index()
        semantic_manifest = self.build_semantic_index() if include_semantic else None
        return {
            "chunks": chunk_manifest,
            "classical": classical_manifest,
            "semantic": semantic_manifest,
        }

    def search(
        self, mode: str, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """Ejecuta una consulta en el modo indicado."""

        normalized_query = query.strip()
        if not normalized_query:
            return []

        top_k = top_k or self.settings.top_k
        if mode == "rag":
            raise ConfigurationError(
                "El modo RAG debe invocarse mediante `answer_rag()` para poder devolver respuesta y fuentes."
            )
        if mode == "semantic":
            return self.search_semantic(normalized_query, top_k=top_k)
        return self.search_classical(normalized_query, top_k=top_k)

    def search_classical(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """Ejecuta búsqueda clásica sobre el índice persistido."""

        top_k = top_k or self.settings.top_k
        _, chunks = self.load_chunks()
        engine = self.load_classical_engine()
        analysis = self.preprocessor.analyze(query)
        return engine.search(query, analysis, chunks, top_k=top_k)

    def search_semantic(
        self, query: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """Ejecuta búsqueda semántica sobre embeddings persistidos."""

        top_k = top_k or self.settings.top_k
        _, chunks = self.load_chunks()
        engine = self.load_semantic_engine()
        embedder = self._semantic_embedder()
        return engine.search(query, chunks, embedder, top_k=top_k)

    def retrieve_rag_sources(
        self, query: str, top_k: int | None = None
    ) -> list[RagSource]:
        """Recupera y fusiona fuentes candidatas para el modo RAG."""

        self._ensure_rag_enabled()

        candidate_top_k = max(
            top_k or self.settings.rag_hybrid_top_k,
            self.settings.rag_hybrid_top_k,
        )
        classical_results = self.search_classical(query, top_k=candidate_top_k)
        semantic_results = self.search_semantic(query, top_k=candidate_top_k)

        retriever = HybridRetriever(
            classical_weight=self.settings.rag_classical_weight,
            semantic_weight=self.settings.rag_semantic_weight,
            rrf_k=self.settings.rag_rrf_k,
        )
        return retriever.combine(
            query=query,
            classical_results=classical_results,
            semantic_results=semantic_results,
            top_k=candidate_top_k,
        )

    def answer_rag(self, query: str, max_sources: int | None = None) -> RagResponse:
        """Genera una respuesta RAG grounded sobre fuentes híbridas."""

        normalized_query = query.strip()
        if not normalized_query:
            raise ConfigurationError("La consulta RAG no puede estar vacía.")

        rag_sources = self.retrieve_rag_sources(normalized_query)
        context_builder = ContextBuilder(
            max_sources=max_sources or self.settings.rag_max_sources,
            max_context_chars=self.settings.rag_max_context_chars,
            max_source_chars=self.settings.rag_max_source_chars,
        )
        context = context_builder.build(rag_sources)
        if not context.sources:
            return RagResponse(
                query=normalized_query,
                answer=(
                    "No he encontrado fragmentos suficientes para responder "
                    "con trazabilidad a partir del corpus recuperado."
                ),
                sources=[],
                references=[],
                model=self.settings.rag_generation_model,
                context="",
                metadata={
                    "mode": "rag",
                    "insufficient_evidence": True,
                    "used_source_count": 0,
                    "available_source_count": 0,
                },
            )

        generator = OllamaRagGenerator(
            host=self.settings.ollama_host,
            model=self.settings.rag_generation_model,
            temperature=self.settings.rag_temperature,
            num_predict=self.settings.rag_num_predict,
            timeout_seconds=self.settings.ollama_timeout_seconds,
        )
        try:
            return generator.generate(normalized_query, context)
        except RagGenerationError:
            raise

    def describe_artifacts(self) -> dict[str, Any]:
        """Devuelve un resumen ligero del estado de los artefactos."""

        return {
            "chunks": self._artifact_summary(self.settings.chunks_path),
            "classical": self._artifact_summary(self.settings.classical_index_path),
            "semantic_manifest": self._artifact_summary(
                self.settings.semantic_manifest_path
            ),
            "semantic_embeddings": {
                "exists": self.settings.semantic_embeddings_path.exists(),
                "path": str(self.settings.semantic_embeddings_path),
            },
        }

    def _artifact_summary(self, path: Path) -> dict[str, Any]:
        summary = {"exists": path.exists(), "path": str(path)}
        if path.exists() and path.suffix == ".json":
            try:
                payload = read_json(path)
            except Exception:
                return summary
            manifest = payload.get("manifest", payload)
            if isinstance(manifest, dict):
                summary["manifest"] = {
                    key: value
                    for key, value in manifest.items()
                    if key
                    in {
                        "built_at",
                        "total_chunks",
                        "total_documents",
                        "model",
                        "dimensions",
                    }
                }
        return summary

    def _load_or_build_chunks(self) -> tuple[dict[str, Any], list[Chunk]]:
        try:
            return self.load_chunks()
        except (ArtifactMissingError, ResourceOutOfDateError):
            logger.info(
                "Los chunks faltaban o estaban desactualizados; se reconstruirán ahora."
            )
            return self.build_chunks()

    def _validate_chunk_manifest(self, manifest: dict[str, Any]) -> None:
        expected_sha = self._corpus_sha256()
        if manifest["corpus_sha256"] != expected_sha:
            raise ResourceOutOfDateError(
                "Los chunks se generaron con una versión distinta del corpus. Ejecuta `build-chunks`."
            )
        if int(manifest["chunk_target_words"]) != self.settings.chunk_target_words:
            raise ResourceOutOfDateError(
                "Los chunks no coinciden con `chunk_target_words`. Ejecuta `build-chunks`."
            )
        if int(manifest["chunk_overlap_words"]) != self.settings.chunk_overlap_words:
            raise ResourceOutOfDateError(
                "Los chunks no coinciden con `chunk_overlap_words`. Ejecuta `build-chunks`."
            )
        if str(manifest["spacy_model"]) != self.settings.spacy_model:
            raise ResourceOutOfDateError(
                "Los chunks no coinciden con el modelo de spaCy configurado. Ejecuta `build-chunks`."
            )

    def _corpus_sha256(self) -> str:
        return sha256_of_file(self.settings.corpus_path)

    def _semantic_embedder(self) -> SpacyVectorEmbedder:
        return SpacyVectorEmbedder(
            nlp=self.preprocessor.nlp,
            model_name=self.settings.spacy_model,
            batch_size=self.settings.semantic_batch_size,
        )

    def _ensure_rag_enabled(self) -> None:
        if not self.settings.rag_enabled:
            raise ConfigurationError(
                "El modo RAG está desactivado en la configuración actual. "
                "Actívalo con `QUIJOTE_RAG__ENABLED=true` o en `settings.toml`."
            )
