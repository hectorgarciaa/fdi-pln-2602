"""Carga tipada de configuración con Dynaconf y variables de entorno."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from dynaconf import Dynaconf

from p4.conf import FRONTMATTER_IDS, STOPWORDS_ES


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Configuración efectiva de la aplicación."""

    repo_root: Path
    project_dir: Path
    corpus_path: Path
    artifacts_dir: Path
    logs_dir: Path
    chunks_path: Path
    classical_index_path: Path
    semantic_embeddings_path: Path
    semantic_manifest_path: Path
    spacy_model: str
    top_k: int
    chunk_target_words: int
    chunk_overlap_words: int
    surface_weight: float
    lemma_weight: float
    semantic_batch_size: int
    semantic_index_version: int
    semantic_title_weight: float
    semantic_body_weight: float
    semantic_original_query_weight: float
    semantic_normalized_query_weight: float
    semantic_lexical_bonus_weight: float
    semantic_rerank_pool_size: int
    ollama_host: str
    ollama_timeout_seconds: float
    rag_enabled: bool
    rag_generation_model: str
    rag_temperature: float
    rag_num_predict: int
    rag_hybrid_top_k: int
    rag_max_sources: int
    rag_max_context_chars: int
    rag_max_source_chars: int
    rag_classical_weight: float
    rag_semantic_weight: float
    rag_rrf_k: int


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """Carga la configuración desde `settings.toml` y `.env`."""

    project_dir = Path(__file__).resolve().parents[1]
    repo_root = project_dir.parent
    load_dotenv(project_dir / ".env", override=False)

    dynasettings = Dynaconf(
        envvar_prefix="QUIJOTE",
        settings_files=[project_dir / "settings.toml"],
        load_dotenv=False,
    )

    artifacts_dir = project_dir / dynasettings.paths.artifacts_dir
    logs_dir = project_dir / dynasettings.paths.logs_dir
    ensure_runtime_directories(artifacts_dir, logs_dir)

    return AppSettings(
        repo_root=repo_root,
        project_dir=project_dir,
        corpus_path=project_dir / dynasettings.paths.corpus_path,
        artifacts_dir=artifacts_dir,
        logs_dir=logs_dir,
        chunks_path=artifacts_dir / dynasettings.paths.chunks_filename,
        classical_index_path=artifacts_dir
        / dynasettings.paths.classical_index_filename,
        semantic_embeddings_path=artifacts_dir
        / dynasettings.paths.semantic_embeddings_filename,
        semantic_manifest_path=artifacts_dir
        / dynasettings.paths.semantic_manifest_filename,
        spacy_model=str(dynasettings.preprocessing.spacy_model),
        top_k=int(dynasettings.search.top_k),
        chunk_target_words=int(dynasettings.chunking.target_words),
        chunk_overlap_words=int(dynasettings.chunking.overlap_words),
        surface_weight=float(dynasettings.search.surface_weight),
        lemma_weight=float(dynasettings.search.lemma_weight),
        semantic_batch_size=int(dynasettings.semantic.batch_size),
        semantic_index_version=int(dynasettings.semantic.index_version),
        semantic_title_weight=float(dynasettings.semantic.title_weight),
        semantic_body_weight=float(dynasettings.semantic.body_weight),
        semantic_original_query_weight=float(
            dynasettings.semantic.original_query_weight
        ),
        semantic_normalized_query_weight=float(
            dynasettings.semantic.normalized_query_weight
        ),
        semantic_lexical_bonus_weight=float(
            dynasettings.semantic.lexical_bonus_weight
        ),
        semantic_rerank_pool_size=int(dynasettings.semantic.rerank_pool_size),
        ollama_host=str(dynasettings.ollama.host),
        ollama_timeout_seconds=float(dynasettings.ollama.timeout_seconds),
        rag_enabled=bool(dynasettings.rag.enabled),
        rag_generation_model=str(dynasettings.rag.generation_model),
        rag_temperature=float(dynasettings.rag.temperature),
        rag_num_predict=int(dynasettings.rag.num_predict),
        rag_hybrid_top_k=int(dynasettings.rag.hybrid_top_k),
        rag_max_sources=int(dynasettings.rag.max_sources),
        rag_max_context_chars=int(dynasettings.rag.max_context_chars),
        rag_max_source_chars=int(dynasettings.rag.max_source_chars),
        rag_classical_weight=float(dynasettings.rag.classical_weight),
        rag_semantic_weight=float(dynasettings.rag.semantic_weight),
        rag_rrf_k=int(dynasettings.rag.rrf_k),
    )


def ensure_runtime_directories(*paths: Path) -> None:
    """Crea directorios de trabajo si no existen."""

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def frontmatter_ids() -> set[str]:
    """Devuelve una copia de los IDs descartados al extraer capítulos."""

    return set(FRONTMATTER_IDS)


def custom_stopwords() -> set[str]:
    """Devuelve las stopwords manuales en formato normalizado."""

    return set(STOPWORDS_ES)
