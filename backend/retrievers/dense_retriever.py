from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.pdf_loader import cargar_pdf
from backend.retrievers.base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Dense retriever local:
    - embeddings con sentence-transformers
    - persistencia en disco con pickle
    - cosine similarity
    - salida compatible con BM25Retriever
    """

    def __init__(
        self,
        pdf_path: str = "data/Doc chatbot.pdf",
        index_path: str = "dense_index.pkl",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.model_name = model_name

        self.model = SentenceTransformer(self.model_name)

        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None

        self._build_or_load()

    def _build_or_load(self) -> None:
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)

            self.texts = data["texts"]
            self.metadatas = data["metadatas"]
            self.embeddings = data["embeddings"]
            return

        docs = cargar_pdf(self.pdf_path)

        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, doc in enumerate(docs):
            text = (getattr(doc, "page_content", "") or "").strip()
            metadata = getattr(doc, "metadata", {}) or {}

            if not text:
                continue

            metadata = dict(metadata)
            metadata.setdefault("chunk_id", idx)

            texts.append(text)
            metadatas.append(metadata)

        if not texts:
            raise ValueError("No se pudieron cargar textos para DenseRetriever.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        self.texts = texts
        self.metadatas = metadatas
        self.embeddings = embeddings

        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                    "embeddings": self.embeddings,
                },
                f,
            )

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.texts) == 0:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            metadata = self.metadatas[idx] or {}
            chunk_id = metadata.get("chunk_id", idx)

            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": self.texts[idx],
                    "metadata": metadata,
                    "score": float(scores[idx]),
                    "retriever": "dense",
                }
            )

        return results