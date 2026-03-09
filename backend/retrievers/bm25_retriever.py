from typing import Any, Dict, List

from backend.baseline_store import BaselineStore
from backend.retrievers.base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, store: BaselineStore):
        self.store = store
        self.store.build_or_load()

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = self.store.search(query, k)

        formatted = []
        for chunk, score in results:
            formatted.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "score": score,
                    "retriever": "bm25",
                }
            )
        return formatted