import os

from backend.baseline_store import BaselineStore
from backend.retrievers.bm25_retriever import BM25Retriever
from backend.retrievers.dense_retriever import DenseRetriever
from backend.retrievers.hybrid_retriever import HybridRetriever


def build_retriever(pdf_path: str = "data/Doc chatbot.pdf"):
    retriever_type = os.getenv("RETRIEVER_TYPE", "bm25").lower()
    alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))

    bm25 = BM25Retriever(BaselineStore(pdf_path=pdf_path))

    if retriever_type == "bm25":
        return bm25

    dense = DenseRetriever(pdf_path=pdf_path)

    if retriever_type == "dense":
        return dense

    if retriever_type == "hybrid":
        return HybridRetriever(bm25, dense, alpha=alpha)

    raise ValueError(f"Retriever no soportado: {retriever_type}")