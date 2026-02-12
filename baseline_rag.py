from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

from backend.baseline_store import BaselineStore, Chunk

NO_EVIDENCE_STD = "No se encontró evidencia suficiente en los documentos."
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _normalize(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", (text or "").lower()))


def _overlap_ratio_qc(question: str, context: str) -> float:
    q = _tokenize(question)
    c = _tokenize(context)
    if not q:
        return 0.0
    return len(q & c) / max(1, len(q))


class BaselineRAG:
    """
    Baseline RAG (sin LLM):
    - Retrieval: BM25 sobre chunks del PDF
    - Answer: extractivo (frases con overlap con la query)
    - Gate: NO_EVIDENCE si score débil o overlap bajo
    - Devuelve contrato estructurado para evaluación
    """

    def __init__(
        self,
        pdf_path: str = "data/Doc chatbot.pdf",
        debug: bool = False,
        k: int = 5,
        min_best_score: float = 0.15,
        min_overlap: float = 0.12,
        max_sentences: int = 6,
    ):
        self.k = k
        self.min_best_score = float(min_best_score)
        self.min_overlap = float(min_overlap)
        self.max_sentences = int(max_sentences)

        self.store = BaselineStore(pdf_path=pdf_path)
        self.store.build_or_load()

        if debug:
            n = len(getattr(self.store, "chunks", []))
            print("CHUNKS:", n)
            if n > 0:
                print("SAMPLE:", self.store.chunks[0].text[:300])

    def _hits_to_retrieved(self, hits: List[Tuple[Chunk, float]]) -> List[Dict[str, Any]]:
        retrieved: List[Dict[str, Any]] = []
        for (c, s) in hits:
            retrieved.append(
                {
                    "chunk_id": getattr(c, "chunk_id", None),
                    "score": float(s),
                    "metadata": getattr(c, "metadata", {}) or {},
                    "text": getattr(c, "text", str(c)),
                }
            )
        return retrieved

    def _best_sentences(self, chunks: List[Chunk], query: str) -> List[str]:
        q_terms = set(re.findall(r"\b\w+\b", (query or "").lower()))
        scored: List[Tuple[int, str]] = []

        for c in chunks:
            txt = _normalize(c.text)
            if not txt:
                continue

            for sent in SENT_SPLIT.split(txt):
                sent = sent.strip()
                if len(sent) < 25:
                    continue
                s_terms = set(re.findall(r"\b\w+\b", sent.lower()))
                overlap = len(q_terms & s_terms)
                if overlap > 0:
                    scored.append((overlap, sent))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[: self.max_sentences]]

    def ask(self, query: str) -> Dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE_STD,
                "no_evidence": True,
                "retrieved": [],
                "used_fallback": False,
                "gate_reason": "empty_question",
                "fuentes": [],
            }

        hits: List[Tuple[Chunk, float]] = self.store.search(query, k=self.k)
        retrieved = self._hits_to_retrieved(hits) if hits else []

        if not hits:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE_STD,
                "no_evidence": True,
                "retrieved": [],
                "used_fallback": False,
                "gate_reason": "no_hits",
                "fuentes": [],
            }

        best_score = float(hits[0][1])
        top_chunks = [c for (c, _s) in hits]

        # Contexto para overlap pregunta↔chunks
        context = " ".join(_normalize(c.text) for c in top_chunks if getattr(c, "text", None))
        overlap = _overlap_ratio_qc(query, context)

        # ✅ Gate por score
        if best_score < self.min_best_score:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE_STD,
                "no_evidence": True,
                "retrieved": retrieved,
                "used_fallback": False,
                "gate_reason": f"bm25_below_min_best_score({best_score:.4f}<{self.min_best_score})",
                "fuentes": [],
            }

        # ✅ Gate por overlap (clave para NO_EVIDENCE)
        if overlap < self.min_overlap:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE_STD,
                "no_evidence": True,
                "retrieved": retrieved,
                "used_fallback": False,
                "gate_reason": f"overlap_below_min_overlap({overlap:.4f}<{self.min_overlap})",
                "fuentes": [],
            }

        # Extractivo
        sentences = self._best_sentences(top_chunks, query)

        if sentences:
            answer = "Según el documento, lo más relevante es:\n" + "\n".join([f"- {s}" for s in sentences])
        else:
            # Pasó gate pero no encontramos frase “directa”
            # (igual devolvemos fuentes, pero sin inventar)
            answer = "Según el documento, encontré fragmentos relacionados, pero no hay una frase directa. Te dejo las fuentes."

        fuentes = [c.text for c in top_chunks]

        return {
            "mode": "baseline",
            "respuesta": answer,
            "no_evidence": False,
            "retrieved": retrieved,
            "used_fallback": False,
            "gate_reason": None,
            "fuentes": fuentes,
        }
