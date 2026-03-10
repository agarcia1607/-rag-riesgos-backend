from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []

        pairs = [(query, doc.get("text", "")) for doc in documents]
        scores = self.model.predict(pairs)

        reranked: List[Dict[str, Any]] = []
        for doc, score in zip(documents, scores):
            item = dict(doc)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]