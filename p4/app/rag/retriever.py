"""Recuperación híbrida para el modo RAG."""

from __future__ import annotations

from dataclasses import dataclass

from p4.app.models import RagSource, SearchResult


@dataclass(slots=True)
class HybridRetriever:
    """Fusiona resultados clásicos y semánticos con una estrategia explicable."""

    classical_weight: float
    semantic_weight: float
    rrf_k: int = 10

    def combine(
        self,
        query: str,
        classical_results: list[SearchResult],
        semantic_results: list[SearchResult],
        top_k: int,
    ) -> list[RagSource]:
        """Deduplica y fusiona rankings de ambos recuperadores."""

        if top_k <= 0:
            return []

        buckets: dict[str, dict[str, object]] = {}
        self._merge_results(
            query=query,
            mode="classical",
            weight=self.classical_weight,
            results=classical_results,
            buckets=buckets,
        )
        self._merge_results(
            query=query,
            mode="semantic",
            weight=self.semantic_weight,
            results=semantic_results,
            buckets=buckets,
        )

        ranked = sorted(
            buckets.values(),
            key=lambda item: (-float(item["score"]), str(item["chunk_id"])),
        )

        sources: list[RagSource] = []
        for index, item in enumerate(ranked[:top_k], start=1):
            explanation = dict(item["explanation"])
            explanation.update(
                {
                    "mode": "rag_source",
                    "query": query,
                    "fusion_method": "weighted normalized score + reciprocal rank fusion",
                    "retrieval_modes": sorted(item["modes"]),
                }
            )
            sources.append(
                RagSource(
                    rank=index,
                    source_id="",
                    chunk_id=str(item["chunk_id"]),
                    document_id=str(item["document_id"]),
                    part=str(item["part"]),
                    title=str(item["title"]),
                    score=round(float(item["score"]), 6),
                    fragment=str(item["fragment"]),
                    text=str(item["text"]),
                    metadata=dict(item["metadata"]),
                    explanation=explanation,
                )
            )
        return sources

    def _merge_results(
        self,
        *,
        query: str,
        mode: str,
        weight: float,
        results: list[SearchResult],
        buckets: dict[str, dict[str, object]],
    ) -> None:
        if not results:
            return

        max_score = max((result.score for result in results), default=0.0)
        for result in results:
            normalized_score = result.score / max_score if max_score > 0 else 0.0
            rank_bonus = 1.0 / (self.rrf_k + result.rank)
            contribution = weight * (normalized_score + rank_bonus)

            entry = buckets.setdefault(
                result.chunk_id,
                {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "part": result.part,
                    "title": result.title,
                    "score": 0.0,
                    "fragment": result.fragment,
                    "text": result.text,
                    "metadata": dict(result.metadata),
                    "explanation": {},
                    "modes": set(),
                    "best_contribution": -1.0,
                },
            )

            entry["score"] = float(entry["score"]) + contribution
            entry["modes"].add(mode)
            explanation = entry["explanation"]
            explanation[f"{mode}_rank"] = result.rank
            explanation[f"{mode}_score"] = round(result.score, 6)
            explanation[f"{mode}_normalized"] = round(normalized_score, 6)
            explanation[f"{mode}_contribution"] = round(contribution, 6)

            if contribution > float(entry["best_contribution"]):
                entry["best_contribution"] = contribution
                entry["fragment"] = result.fragment
                entry["text"] = result.text
                entry["metadata"] = dict(result.metadata)
