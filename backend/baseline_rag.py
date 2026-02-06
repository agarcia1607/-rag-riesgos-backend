import re
from typing import Dict, Any, List
from backend.baseline_store import BaselineStore, Chunk


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _normalize(text: str) -> str:
    # Normalización mínima para PDFs (espacios raros, saltos, etc.)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class BaselineRAG:
    """
    Baseline RAG (sin tokens):
    - Retrieval: BM25 sobre chunks del PDF
    - "Generation": respuesta extractiva (selección de frases relevantes)
    """

    def __init__(self, pdf_path: str = "data/Doc chatbot.pdf", debug: bool = False):
        self.store = BaselineStore(pdf_path=pdf_path)
        self.store.build_or_load()

        if debug:
            n = len(getattr(self.store, "chunks", []))
            print("CHUNKS:", n)
            if n > 0:
                print("SAMPLE:", self.store.chunks[0].text[:300])

    def _best_sentences(self, chunks: List[Chunk], query: str, max_sentences: int = 6) -> List[str]:
        q_terms = set(re.findall(r"\b\w+\b", query.lower()))
        scored = []

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
        return [s for _, s in scored[:max_sentences]]

    def ask(self, query: str, k: int = 5, min_score: float = 0.0) -> Dict[str, Any]:
        """
        Responde con un baseline extractivo:
        - Recupera top-k chunks con BM25
        - Extrae frases con mayor overlap de términos con la consulta

        Nota:
        min_score = 0.0 para no bloquear respuestas en PDFs donde BM25 da scores bajos.
        """
        hits = self.store.search(query, k=k)

        if not hits:
            return {
                "respuesta": "No encontré fragmentos relevantes para esa consulta en el documento.",
                "fuentes": []
            }

        top_chunks = [c for c, _ in hits]
        sentences = self._best_sentences(top_chunks, query)

        best_score = hits[0][1] if hits else 0.0
        low_confidence = best_score < 0.1

        if sentences:
            header = "Según el documento, lo más relevante es:"
            if low_confidence:
                header = "No estoy 100% seguro, pero encontré fragmentos relacionados. Lo más relevante es:"
            answer = header + "\n" + "\n".join([f"- {s}" for s in sentences])
        else:
            answer = "Encontré fragmentos relacionados, pero no hay una frase directa. Te dejo las fuentes."

        fuentes = [c.text for c in top_chunks]
        return {"respuesta": answer, "fuentes": fuentes}
