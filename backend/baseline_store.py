import os
import re
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

from backend.pdf_loader import cargar_pdf
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


@dataclass
class Chunk:
    chunk_id: int
    text: str
    metadata: Dict[str, Any]


class BaselineStore:
    def __init__(self, pdf_path: str = "Doc chatbot.pdf", persist_path: str = "baseline_bm25.pkl"):
        self.pdf_path = pdf_path
        self.persist_path = persist_path

        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: BM25Okapi | None = None

    def build_or_load(self) -> None:
        # Load if exists
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as f:
                obj = pickle.load(f)
            self.chunks = obj["chunks"]
            self.corpus_tokens = obj["corpus_tokens"]
            self.bm25 = BM25Okapi(self.corpus_tokens)
            return

        # Build from PDF
        docs = cargar_pdf(self.pdf_path)
        self.chunks = [
            Chunk(chunk_id=i, text=d.page_content, metadata=getattr(d, "metadata", {}) or {})
            for i, d in enumerate(docs)
        ]
        self.corpus_tokens = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

        # Persist
        with open(self.persist_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "corpus_tokens": self.corpus_tokens}, f)

    def search(self, query: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Devuelve top-k hits BM25.
        FIX CLAVE: si BM25 no encuentra señal (todos los scores == 0),
        devolvemos [] para activar 'no_hits' y permitir abstención.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 not initialized. Call build_or_load() first.")

        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)

        # ✅ Si todos son 0.0 (o no positivos), NO devolvemos basura: devolvemos []
        ranked = [(i, float(s)) for i, s in enumerate(scores) if float(s) > 0.0]
        if not ranked:
            return []

        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[:k]

        return [(self.chunks[i], s) for i, s in ranked]