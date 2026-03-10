from __future__ import annotations

from typing import Any, Dict, List

from backend.retrievers.base_retriever import BaseRetriever
from backend.retrievers.reranker import Reranker


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever:
    - combina BM25 y Dense
    - fusiona por chunk_id
    - normaliza scores por ranking simple
    - rerankea candidatos con cross-encoder
    - devuelve formato estándar
    """

    def __init__(
        self,
        bm25_retriever: Any,
        dense_retriever: Any,
        alpha: float = 0.5,
        initial_k: int = 15,
        rerank_top_k: int = 5,
        use_reranker: bool = True,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.alpha = alpha
        self.initial_k = initial_k
        self.rerank_top_k = rerank_top_k
        self.use_reranker = use_reranker
        self.reranker = Reranker() if use_reranker else None

    def _rank_fusion_scores(self, items: List[Dict[str, Any]]) -> Dict[Any, float]:
        """
        Convierte ranking en score tipo Reciprocal Rank Fusion simplificado.
        score = 1 / (rank + 1)
        """
        scores: Dict[Any, float] = {}
        for rank, item in enumerate(items):
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                continue
            scores[chunk_id] = 1.0 / (rank + 1)
        return scores

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        candidate_k = max(self.initial_k, k)

        bm25_results = self.bm25_retriever.retrieve(query, k=max(candidate_k * 2, 10))
        dense_results = self.dense_retriever.retrieve(query, k=max(candidate_k * 2, 10))

        bm25_rank_scores = self._rank_fusion_scores(bm25_results)
        dense_rank_scores = self._rank_fusion_scores(dense_results)

        merged: Dict[Any, Dict[str, Any]] = {}

        for item in bm25_results:
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                continue

            merged[chunk_id] = {
                **item,
                "bm25_score": bm25_rank_scores.get(chunk_id, 0.0),
                "dense_score": 0.0,
            }

        for item in dense_results:
            chunk_id = item.get("chunk_id")
            if chunk_id is None:
                continue

            if chunk_id not in merged:
                merged[chunk_id] = {
                    **item,
                    "bm25_score": 0.0,
                    "dense_score": dense_rank_scores.get(chunk_id, 0.0),
                }
            else:
                merged[chunk_id]["dense_score"] = dense_rank_scores.get(chunk_id, 0.0)

        combined: List[Dict[str, Any]] = []
        for item in merged.values():
            bm25_score = item.get("bm25_score", 0.0)
            dense_score = item.get("dense_score", 0.0)

            final_score = self.alpha * bm25_score + (1.0 - self.alpha) * dense_score

            row = dict(item)
            row["score"] = float(final_score)
            row["retriever"] = "hybrid"
            combined.append(row)

        combined.sort(key=lambda x: x["score"], reverse=True)

        candidates = combined[:candidate_k]

        if self.use_reranker and self.reranker is not None:
            try:
                final_k = min(k, self.rerank_top_k)
                reranked = self.reranker.rerank(query, candidates, top_k=final_k)

                for item in reranked:
                    item["retriever"] = "hybrid_rerank"

                return reranked
            except Exception as e:
                print(f"[WARN] Reranker failed, using hybrid ranking only: {e}")

        return candidates[:k]