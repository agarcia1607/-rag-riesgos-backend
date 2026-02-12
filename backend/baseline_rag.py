from __future__ import annotations

from typing import Dict, Any, List, Tuple
import logging

from backend.baseline_store import BaselineStore

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."


def _has_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


class BaselineRAG:
    """
    Baseline (extractivo, reproducible):
    - Retrieval: BM25 (BaselineStore)
    - Respuesta: extractiva / bullet points desde los chunks

    Gates:
    - empty_question
    - no_hits
    - required_topic_missing(cyber)
    - external_entity_question_gate
    """

    def __init__(
        self,
        pdf_path: str,
        debug: bool = False,
        k: int = 5,
    ):
        self.pdf_path = pdf_path
        self.debug = debug
        self.k = k

        # ✅ Inicializa BM25 store y construye/carga índice
        self.store = BaselineStore(pdf_path=self.pdf_path)
        self.store.build_or_load()

        logger.info("🟦 BaselineRAG listo (BM25 inicializado).")

    def _hits_to_retrieved(self, hits: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
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

    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "empty_question",
            }

        hits: List[Tuple[Any, float]] = self.store.search(question, k=self.k)
        if not hits:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_hits",
            }

        retrieved = self._hits_to_retrieved(hits)
        fuentes = [r["text"] for r in retrieved]

        q = question.lower()
        context_joined = " ".join(fuentes).lower()

        # =========================
        # ✅ GATE: cyber
        # =========================
        cyber_q_terms = ["ciber", "cibern", "cyber", "ransom", "malware", "phishing", "ddos", "hack"]
        cyber_ctx_terms = cyber_q_terms + ["intrusión", "intrusion", "ataque", "ataques"]
        if _has_any(q, cyber_q_terms) and not _has_any(context_joined, cyber_ctx_terms):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "required_topic_missing(cyber)",
            }

        # =========================
        # ✅ GATE: entidad externa (CEO, etc.)
        # =========================
        external_terms = ["ceo", "director general", "presidente", "owner", "propietario", "gerente general"]
        if _has_any(q, external_terms):
            # Si el PDF no menciona explícitamente un nombre en el contexto top-k,
            # declaramos no_evidence. Esto evita alucinación.
            # (Puedes endurecerlo con regex de nombres propios si quieres.)
            if not _has_any(context_joined, [" ceo", "director general", "presidente"]):
                return {
                    "mode": "baseline",
                    "respuesta": NO_EVIDENCE,
                    "fuentes": fuentes,
                    "retrieved": retrieved,
                    "no_evidence": True,
                    "used_fallback": False,
                    "gate_reason": "external_entity_question_gate",
                }

        # =========================
        # Extractivo simple (no inventa)
        # =========================
        bullets = []
        for r in retrieved[: min(5, len(retrieved))]:
            t = (r["text"] or "").strip()
            if not t:
                continue
            bullets.append(f"- {t}")

        respuesta = "Según el documento, lo más relevante es:\n" + "\n".join(bullets)

        return {
            "mode": "baseline",
            "respuesta": respuesta,
            "fuentes": fuentes,
            "retrieved": retrieved,
            "no_evidence": False,
            "used_fallback": False,
            "gate_reason": None,
        }
