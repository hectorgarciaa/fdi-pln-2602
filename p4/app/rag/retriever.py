"""Recuperación híbrida para el modo RAG."""

from __future__ import annotations

from dataclasses import dataclass
import re

from p4.conf import STOPWORDS_ES
from p4.app.models import RagSource, SearchResult
from p4.app.utils import normalize_token


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(slots=True)
class HybridRetriever:
    """Fusiona resultados clásicos y semánticos con una estrategia explicable."""

    classical_weight: float
    semantic_weight: float
    rrf_k: int = 10
    min_score_ratio: float = 0.96
    semantic_only_zero_overlap_penalty: float = 0.3

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

        query_terms = _query_terms(query)
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
        self._tighten_candidates(buckets=buckets, query_terms=query_terms)

        ranked = sorted(
            self._filter_by_score_ratio(buckets.values()),
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
                    "lexical_overlap_ratio": round(
                        float(item.get("lexical_overlap_ratio", 0.0)), 6
                    ),
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

    def _tighten_candidates(
        self, *, buckets: dict[str, dict[str, object]], query_terms: set[str]
    ) -> None:
        for entry in buckets.values():
            overlap_ratio = _lexical_overlap_ratio(
                query_terms=query_terms,
                title=str(entry["title"]),
                text=str(entry["text"]),
            )
            entry["lexical_overlap_ratio"] = overlap_ratio
            explanation = entry["explanation"]
            explanation["lexical_overlap_ratio"] = round(overlap_ratio, 6)

            modes = set(entry["modes"])
            if (
                modes == {"semantic"}
                and overlap_ratio == 0.0
                and self.semantic_only_zero_overlap_penalty > 0
            ):
                original_score = float(entry["score"])
                penalized_score = original_score * (
                    1.0 - self.semantic_only_zero_overlap_penalty
                )
                entry["score"] = max(penalized_score, 0.0)
                explanation["semantic_only_penalty_applied"] = True
                explanation["semantic_only_penalty"] = round(
                    self.semantic_only_zero_overlap_penalty, 6
                )
                explanation["score_before_penalty"] = round(original_score, 6)
                explanation["score_after_penalty"] = round(float(entry["score"]), 6)
            else:
                explanation["semantic_only_penalty_applied"] = False

    def _filter_by_score_ratio(self, entries: object) -> list[dict[str, object]]:
        ranked_entries = list(entries)
        if not ranked_entries:
            return []

        best_score = max(float(item["score"]) for item in ranked_entries)
        if best_score <= 0:
            return ranked_entries

        cutoff = best_score * self.min_score_ratio
        filtered = [item for item in ranked_entries if float(item["score"]) >= cutoff]
        if filtered:
            return filtered
        return sorted(
            ranked_entries,
            key=lambda item: (-float(item["score"]), str(item["chunk_id"])),
        )[:1]


def _query_terms(query: str) -> set[str]:
    stopwords = {normalize_token(word) for word in STOPWORDS_ES}
    terms: set[str] = set()
    for raw_token in WORD_RE.findall(query):
        token = normalize_token(raw_token)
        if len(token) < 2 or token in stopwords:
            continue
        terms.add(token)
    return terms


def _lexical_overlap_ratio(*, query_terms: set[str], title: str, text: str) -> float:
    if not query_terms:
        return 0.0

    candidate_terms: set[str] = set()
    for raw_token in WORD_RE.findall(f"{title} {text}"):
        token = normalize_token(raw_token)
        if len(token) >= 2:
            candidate_terms.add(token)

    if not candidate_terms:
        return 0.0

    overlap = query_terms & candidate_terms
    return len(overlap) / len(query_terms)
