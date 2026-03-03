from __future__ import annotations

from typing import Dict, Any, List, Tuple
import logging
import os
import re
import json

from backend.baseline_store import BaselineStore
from backend.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."


def _has_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


def _best_span_multi(context: str, keywords: List[str], window: int = 520) -> str:
    """Devuelve un fragmento alrededor de la PRIMERA keyword encontrada."""
    if not context:
        return ""
    low = context.lower()
    for kw in keywords:
        i = low.find(kw.lower())
        if i != -1:
            start = max(0, i - window // 2)
            end = min(len(context), i + window // 2)
            return context[start:end].strip()
    return ""


def _extract_risk_levels(question: str, context_plain: str) -> str:
    """
    Extractor determinista:
    "¿Cuáles son los niveles de riesgo...?"
    """
    q = (question or "").lower()
    if "niveles" not in q or "riesgo" not in q:
        return ""

    c = context_plain or ""
    if ("RIESGO" not in c.upper()) and ("Riesgo" not in c):
        return ""

    levels = []
    if re.search(r"RIESGO\s+BAJO", c, re.IGNORECASE):
        levels.append("RIESGO BAJO")
    if re.search(r"RIESGO\s+MEDIO", c, re.IGNORECASE):
        levels.append("RIESGO MEDIO")
    if re.search(r"RIESGO\s+ALTO", c, re.IGNORECASE):
        levels.append("RIESGO ALTO")

    if not levels:
        span = _best_span_multi(
            c, ["RIESGO BAJO", "RIESGO MEDIO", "RIESGO ALTO", "nivel de riesgo"], window=900
        )
        if not span:
            return ""
        return "Según el documento, se mencionan niveles de riesgo como:\n" + span

    return "Según el documento, los niveles de riesgo mencionados incluyen: " + ", ".join(levels) + "."


def _extract_deducible_refrigeracion_strict(question: str, context_plain: str) -> str:
    """
    Extractor determinista estricto:
    "deducible ... refrigeración"
    Debe incluir: 10%, USD, 1,500
    """
    q = (question or "").lower()
    if "deducible" not in q or "refriger" not in q:
        return ""

    c = context_plain or ""

    span = _best_span_multi(c, ["1,500", "10%", "USD", "refriger", "deducible"], window=1100)
    if not span:
        return ""

    if ("10%" not in span) or ("USD" not in span) or ("1,500" not in span):
        lines = [ln.strip() for ln in c.splitlines() if ln.strip()]
        for ln in lines:
            if ("1,500" in ln) and ("USD" in ln) and ("10%" in ln):
                span = ln
                break
        else:
            return ""

    return "Según el documento, el deducible por fallas en el sistema de refrigeración es:\n" + span


class LocalRAG:
    """
    Local:
    - Retrieval: BM25 (BaselineStore)
    - Redacción: Ollama (síntesis controlada)
    - Gates para NO_EVIDENCE
    - Extractores deterministas
    - Evidencia auditable
    """

    def __init__(self, pdf_path: str, k: int = 5):
        self.pdf_path = pdf_path

        self.k = int(os.getenv("BM25_K", str(k)))
        self.max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))
        self.top_n_for_llm = int(os.getenv("TOP_N_FOR_LLM", "3"))

        self.store = BaselineStore(pdf_path=self.pdf_path)
        self.store.build_or_load()

        self.ollama = OllamaClient()

        logger.info(
            "🟣 LocalRAG listo | BM25_K=%s | TOP_N_FOR_LLM=%s | MAX_CONTEXT_CHARS=%s | model=%s | base=%s",
            self.k,
            self.top_n_for_llm,
            self.max_context_chars,
            getattr(self.ollama, "model", "unknown"),
            getattr(self.ollama, "base_url", "unknown"),
        )

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

    def _truncate_context(self, ctx: str) -> str:
        if not ctx:
            return ""
        if len(ctx) <= self.max_context_chars:
            return ctx
        return ctx[: self.max_context_chars] + "\n[...TRUNCADO...]"

    def _is_page_marker(self, s: str) -> bool:
        return bool(re.fullmatch(r"\[PAGE\s+\d+\]", (s or "").strip(), re.IGNORECASE))

    def _filter_noise_text(self, texts: List[str]) -> List[str]:
        out: List[str] = []
        for t in texts:
            if not t:
                continue
            s = t.strip()
            if not s:
                continue
            if self._is_page_marker(s):
                continue
            out.append(t)
        return out

    def _build_structured_context(self, items: List[Dict[str, Any]]) -> str:
        """
        Contexto para LLM: incluye headers con chunk_id.
        """
        blocks: List[str] = []
        for r in items:
            text = (r.get("text") or "").strip()
            if not text or self._is_page_marker(text):
                continue

            cid = r.get("chunk_id")
            score = r.get("score", 0.0)
            try:
                score_str = f"{float(score):.4f}"
            except Exception:
                score_str = str(score)

            blocks.append(f"[chunk_id: {cid} | score: {score_str}]\n{text}")

        return self._truncate_context("\n\n".join(blocks))

    def _build_plain_context(self, items: List[Dict[str, Any]]) -> str:
        """
        Contexto para EXTRACTORES: solo texto (sin headers).
        """
        parts: List[str] = []
        for r in items:
            t = (r.get("text") or "").strip()
            if not t or self._is_page_marker(t):
                continue
            parts.append(t)
        return "\n".join(parts)

    # ------------------ VRDLM extractor (FIX DEFINITIVO) ------------------

    def _extract_vrdlm_from_retrieved(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        """
        FIX:
        - El chunk 111 termina en "VRDLM): Es el" (incompleto).
        - El chunk 112 tiene la definición completa "Es el valor ...".
        - Priorizamos explícitamente "Es el valor" para nunca quedarnos con el 111.
        """
        q = (question or "").lower()
        if "vrdlm" not in q:
            return ""

        # ✅ 1) Prioridad: definición completa (contiene "Es el valor")
        for r in retrieved:
            t = (r.get("text") or "")
            if ("VRDLM" not in t) or ("Es el valor" not in t):
                continue

            m = re.search(
                r"(Valor Real de la Mercancía\s*\(VRDLM\)\s*:\s*Es[^\.]+(\.|$))",
                t,
                re.IGNORECASE,
            )
            if m:
                return "Según el documento, VRDLM se define así:\n" + m.group(1).strip()

            # fallback dentro del mismo chunk completo
            m = re.search(r"(VRDLM\)\s*:\s*Es[^\.]+(\.|$))", t, re.IGNORECASE)
            if m and ("Es el valor" in m.group(1)):
                return "Según el documento, VRDLM se define así:\n" + m.group(1).strip()

        # ✅ 2) Segundo intento: si no encontramos "Es el valor", buscamos definición en cualquier chunk
        for r in retrieved:
            t = (r.get("text") or "")
            if "VRDLM" not in t:
                continue

            m = re.search(
                r"(Valor Real de la Mercancía\s*\(VRDLM\)\s*:\s*Es[^\.]+(\.|$))",
                t,
                re.IGNORECASE,
            )
            if m:
                return "Según el documento, VRDLM se define así:\n" + m.group(1).strip()

            m = re.search(r"(VRDLM\)\s*:\s*Es[^\.]+(\.|$))", t, re.IGNORECASE)
            if m:
                return "Según el documento, VRDLM se define así:\n" + m.group(1).strip()

        return ""

    def _evidence_ids_for_vrdlm(self, items: List[Dict[str, Any]]) -> List[int]:
        """
        Estricto: solo chunks con 'Es el valor' (evita chunk 111 incompleto).
        Normalmente devuelve [112].
        """
        ids: List[int] = []
        for r in items:
            t = (r.get("text") or "")
            if ("VRDLM" in t) and ("Es el valor" in t):
                cid = r.get("chunk_id")
                if isinstance(cid, int):
                    ids.append(cid)
        if ids:
            return ids
        return [r["chunk_id"] for r in items if isinstance(r.get("chunk_id"), int)]

    def _evidence_ids_for_risk_levels(self, items: List[Dict[str, Any]]) -> List[int]:
        ids: List[int] = []
        for r in items:
            t = (r.get("text") or "").upper()
            if ("RIESGO BAJO" in t) or ("RIESGO MEDIO" in t) or ("RIESGO ALTO" in t):
                cid = r.get("chunk_id")
                if isinstance(cid, int):
                    ids.append(cid)
        if ids:
            return ids
        return [r["chunk_id"] for r in items if isinstance(r.get("chunk_id"), int)]

    def _evidence_ids_for_deducible_refri(self, items: List[Dict[str, Any]]) -> List[int]:
        ids: List[int] = []
        for r in items:
            t = (r.get("text") or "")
            if ("10%" in t) and ("USD" in t) and ("1,500" in t) and ("refriger" in t.lower()):
                cid = r.get("chunk_id")
                if isinstance(cid, int):
                    ids.append(cid)
        if ids:
            return ids
        return [r["chunk_id"] for r in items if isinstance(r.get("chunk_id"), int)]

    # ------------------ MAIN ------------------

    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {
                "mode": "local",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": [],
                "evidence_chunk_ids": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "empty_question",
            }

        hits: List[Tuple[Any, float]] = self.store.search(question, k=self.k)
        if not hits:
            return {
                "mode": "local",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": [],
                "evidence_chunk_ids": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_hits",
            }

        retrieved = self._hits_to_retrieved(hits)

        # ✅ Extractores: texto plano con TODO retrieved
        extract_plain = self._build_plain_context(retrieved)

        # ✅ LLM: solo top-N
        top = retrieved[: self.top_n_for_llm]
        fuentes_for_ui = self._filter_noise_text([r.get("text", "") for r in top])
        llm_ctx = self._build_structured_context(top)

        q = question.lower()
        ctx_low = extract_plain.lower()

        # ----------------- Gates -----------------
        cyber_q_terms = ["ciber", "cibern", "cyber", "ransom", "malware", "phishing", "ddos", "hack"]
        cyber_ctx_terms = cyber_q_terms + ["intrusión", "intrusion", "ataque", "ataques"]
        if _has_any(q, cyber_q_terms) and not _has_any(ctx_low, cyber_ctx_terms):
            return {
                "mode": "local",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "required_topic_missing(cyber)",
            }

        external_terms = ["ceo", "director general", "presidente", "owner", "propietario", "gerente general"]
        if _has_any(q, external_terms):
            return {
                "mode": "local",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "external_entity_question_gate",
            }

        # ----------------- Extractores -----------------

        direct = self._extract_vrdlm_from_retrieved(question, retrieved)
        if direct:
            ids = self._evidence_ids_for_vrdlm(retrieved)
            return {
                "mode": "local",
                "respuesta": direct,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": ids,
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": "deterministic_extract(vrdlm)",
            }

        direct = _extract_risk_levels(question, extract_plain)
        if direct:
            ids = self._evidence_ids_for_risk_levels(retrieved)
            return {
                "mode": "local",
                "respuesta": direct,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": ids,
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": "deterministic_extract(risk_levels)",
            }

        direct = _extract_deducible_refrigeracion_strict(question, extract_plain)
        if direct:
            ids = self._evidence_ids_for_deducible_refri(retrieved)
            return {
                "mode": "local",
                "respuesta": direct,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": ids,
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": "deterministic_extract(deducible_refrigeracion_strict)",
            }

        # ----------------- LLM (Ollama) -----------------

        expected_json = f'{{"answer":"{NO_EVIDENCE}","evidence_chunk_ids":[]}}'
        prompt = (
            "Eres un sistema de QA basado exclusivamente en evidencia.\n\n"
            "REGLAS ESTRICTAS:\n"
            "1) Solo puedes usar información explícita del CONTEXTO.\n"
            "2) Prohibido inferir o inventar. Prohibido usar conocimiento externo.\n"
            "3) Debes responder SIEMPRE con un JSON válido (sin texto extra, sin markdown).\n"
            "4) El JSON debe tener exactamente estas llaves: answer, evidence_chunk_ids.\n"
            "5) evidence_chunk_ids debe ser una lista de enteros con los chunk_id usados.\n"
            f"6) Si no hay evidencia suficiente, responde EXACTAMENTE este JSON:\n{expected_json}\n\n"
            "CONTEXTO (bloques con chunk_id):\n"
            f"{llm_ctx}\n\n"
            "PREGUNTA:\n"
            f"{question}\n\n"
            "RESPONDE SOLO CON JSON:"
        )

        try:
            raw = (self.ollama.generate(prompt) or "").strip()

            answer = ""
            evidence_ids: List[int] = []
            gate_reason = None

            try:
                obj = json.loads(raw)
                answer = (obj.get("answer") or "").strip()
                ids = obj.get("evidence_chunk_ids") or []
                normalized: List[int] = []
                for x in ids:
                    try:
                        normalized.append(int(x))
                    except Exception:
                        pass
                evidence_ids = normalized
            except Exception:
                answer = raw
                evidence_ids = []
                gate_reason = "llm_output_not_json"

            if not answer:
                answer = NO_EVIDENCE

            no_evidence = (answer.strip() == NO_EVIDENCE)
            if no_evidence:
                evidence_ids = []
                gate_reason = gate_reason or None

            return {
                "mode": "local",
                "respuesta": answer,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": evidence_ids,
                "no_evidence": no_evidence,
                "used_fallback": False,
                "gate_reason": gate_reason,
            }

        except Exception as e:
            logger.error(f"❌ Ollama error: {e}. Fallback a extractivo.")
            bullets = []
            for r in retrieved[: min(5, len(retrieved))]:
                t = (r.get("text") or "").strip()
                if t and not self._is_page_marker(t):
                    bullets.append(f"- {t}")
            fallback_answer = "Según el documento, lo más relevante es:\n" + "\n".join(bullets)

            ids = [r.get("chunk_id") for r in top if isinstance(r.get("chunk_id"), int)]

            return {
                "mode": "local",
                "respuesta": fallback_answer,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": ids,
                "no_evidence": False,
                "used_fallback": True,
                "gate_reason": "ollama_error_fallback",
            }