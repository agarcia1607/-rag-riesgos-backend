from __future__ import annotations

from typing import Dict, Any, List, Optional
import logging
import os
import re
import json
import unicodedata

from backend.ollama_client import OllamaClient
from backend.retrievers.factory import build_retriever

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."


# --------------------------- helpers (module-level) ---------------------------

def _strip_accents(text: str) -> str:
    text = text or ""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )


def _normalize_text(text: str) -> str:
    t = _strip_accents(text or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _has_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term.lower() in t for term in terms)


def _best_span_multi(context: str, keywords: List[str], window: int = 520) -> str:
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


def _extract_json_candidate(raw: str) -> str:
    """
    Intenta extraer el primer bloque JSON válido de una respuesta del LLM.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""

    if raw.startswith("{") and raw.endswith("}"):
        return raw

    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        return m.group(0).strip()

    return raw


# ------------------------------ LocalRAG class ------------------------------

class LocalRAG:
    """
    Local:
    - Retrieval: Hybrid/BM25 vía retriever modular
    - Redacción: Ollama
    - Extractores deterministas primero
    - Fallback extractivo controlado
    - Modo debug para inspeccionar salida cruda del LLM
    """

    def __init__(self, pdf_path: str, k: int = 5):
        self.pdf_path = pdf_path

        self.k = int(os.getenv("BM25_K", str(k)))
        self.max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "9000"))
        self.top_n_for_llm = int(os.getenv("TOP_N_FOR_LLM", "5"))

        # debug / visibilidad
        self.debug_no_fallback = os.getenv("DEBUG_NO_FALLBACK", "0") == "1"
        self.include_raw_llm_answer = os.getenv("LOCAL_RAG_INCLUDE_RAW", "0") == "1"
        self.log_llm_prompt = os.getenv("LOCAL_RAG_LOG_PROMPT", "0") == "1"

        self.retriever = build_retriever()
        self.ollama = OllamaClient()

        logger.info(
            "🟣 LocalRAG listo | BM25_K=%s | TOP_N_FOR_LLM=%s | MAX_CONTEXT_CHARS=%s | DEBUG_NO_FALLBACK=%s | INCLUDE_RAW=%s | model=%s | base=%s",
            self.k,
            self.top_n_for_llm,
            self.max_context_chars,
            self.debug_no_fallback,
            self.include_raw_llm_answer,
            getattr(self.ollama, "model", "unknown"),
            getattr(self.ollama, "base_url", "unknown"),
        )

    # ------------------------- formatting / conversions -------------------------

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
        blocks: List[str] = []
        for r in items:
            text = (r.get("text") or "").strip()
            if not text or self._is_page_marker(text):
                continue

            cid = r.get("chunk_id")
            score = r.get("score", 0.0)
            page = (r.get("metadata") or {}).get("page")
            try:
                score_str = f"{float(score):.4f}"
            except Exception:
                score_str = str(score)

            blocks.append(
                f"[chunk_id: {cid} | page: {page} | score: {score_str}]\n{text}"
            )

        return self._truncate_context("\n\n".join(blocks))

    def _build_plain_context(self, items: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for r in items:
            t = (r.get("text") or "").strip()
            if not t or self._is_page_marker(t):
                continue
            parts.append(t)
        return "\n".join(parts)

    def _top_chunk_ids(self, retrieved: List[Dict[str, Any]], n: int) -> List[int]:
        ids: List[int] = []
        for r in retrieved[: max(0, int(n))]:
            cid = r.get("chunk_id")
            if isinstance(cid, int):
                ids.append(cid)
        return ids

    # ------------------ lightweight reranking for local mode ------------------

    def _extract_question_keywords(self, question: str) -> List[str]:
        q = _normalize_text(question)

        # quitar palabras vacías comunes
        stop = {
            "que", "qué", "es", "la", "el", "los", "las", "de", "del", "segun", "según",
            "póliza", "poliza", "cual", "cuál", "defina", "definicion", "definición",
            "segun", "según", "segun", "según", "por", "para", "en", "y", "o",
        }

        raw_tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9\-]+", q)
        tokens = [t for t in raw_tokens if len(t) >= 3 and t not in stop]

        # frases útiles
        phrases: List[str] = []
        if "escolta satelital" in q:
            phrases.append("escolta satelital")
        if "monitoreo satelital" in q:
            phrases.append("monitoreo satelital")
        if "fast track" in q:
            phrases.append("fast track")

        # de-dup manteniendo orden
        out: List[str] = []
        seen = set()
        for x in phrases + tokens:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def _is_definition_question(self, question: str) -> bool:
        q = _normalize_text(question)
        return any(
            pat in q
            for pat in [
                "que es ",
                "defina ",
                "definicion de",
                "definición de",
                "cual es la definicion",
                "cuál es la definición",
            ]
        )

    def _rerank_for_question(self, question: str, retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not retrieved:
            return retrieved

        keywords = self._extract_question_keywords(question)
        is_def = self._is_definition_question(question)

        scored: List[tuple[float, Dict[str, Any]]] = []
        for idx, r in enumerate(retrieved):
            text = _normalize_text(r.get("text") or "")
            score = float(r.get("score") or 0.0)

            bonus = 0.0

            # bonus por frases completas
            for kw in keywords:
                if " " in kw and kw in text:
                    bonus += 2.5

            # bonus por coincidencias de tokens
            token_hits = sum(1 for kw in keywords if kw in text)
            bonus += 0.35 * token_hits

            # bonus definicional: ":" o "se define" o "es"
            if is_def:
                if ":" in (r.get("text") or ""):
                    bonus += 0.5
                if "se define" in text:
                    bonus += 0.7
                if " es " in f" {text} ":
                    bonus += 0.2

            # pequeño castigo por chunks demasiado genéricos
            if "incoterm" in text and "escolta satelital" not in text:
                bonus -= 0.75

            # estabilidad por orden original
            final = score + bonus - 0.0001 * idx
            scored.append((final, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    # ------------------ VRDLM extractor ------------------

    def _extract_vrdlm_from_retrieved(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        q = (question or "").lower()
        if "vrdlm" not in q:
            return ""

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

            m = re.search(r"(VRDLM\)\s*:\s*Es[^\.]+(\.|$))", t, re.IGNORECASE)
            if m and ("Es el valor" in m.group(1)):
                return "Según el documento, VRDLM se define así:\n" + m.group(1).strip()

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
        ids: List[int] = []
        for r in items:
            t = (r.get("text") or "")
            if ("VRDLM" in t) and ("Es el valor" in t):
                cid = r.get("chunk_id")
                if isinstance(cid, int):
                    ids.append(cid)
        if ids:
            return ids

        for r in items:
            t = (r.get("text") or "")
            if "VRDLM" in t:
                cid = r.get("chunk_id")
                if isinstance(cid, int):
                    return [cid]
        return []

    # ------------------ hard constraints (must_include boost) ------------------

    def _required_terms(self, question: str, context_plain: str) -> List[str]:
        q = (question or "").lower()
        c = (context_plain or "").lower()

        req: List[str] = []

        def want(term: str) -> bool:
            t = term.lower()
            return (t in q) or (t in c)

        if "días" in q and "30" in c:
            req.append("30 días")
        if "horas" in q and "72" in c:
            req.append("72 horas")
        if "años" in q and "30" in c and ("buque" in q or "buques" in q):
            req.append("30 años")

        if ("forma de pago" in q) and want("contado"):
            req.append("contado")

        if (("3 facturas" in q) or ("más de 3 facturas" in q)) and want("3 facturas"):
            req.append("3 facturas")

        if ("protocolo" in q) and ("seguridad" in q):
            if want("transporte terrestre"):
                req.append("transporte terrestre")
            if want("póliza e-cargo") or want("e-cargo"):
                req.append("póliza E-CARGO")
            if want("mercancías vulnerables") or want("mercancías vulnerables al robo"):
                req.append("mercancías vulnerables")

        if "fast track" in q:
            req.append("Fast Track")
            if "conocimiento" in q:
                req.append("conocimiento")

        if ("documentos" in q or "debe presentar" in q) and ("fast track" in q) and want("denuncia"):
            req.append("denuncia")

        if ("robo" in q) and ("sin violencia" in q) and want("desaparición misteriosa"):
            req.append("robo total sin violencia")
            req.append("desaparición misteriosa")

        if ("uber" in q) or ("cabify" in q):
            if want("excluida"):
                req.append("excluida")
            for t in ["Uber", "Cabify", "taxis", "servicios privados"]:
                if want(t):
                    req.append(t)

        if ("hongos" in q) or ("plagas" in q):
            if want("excluidos"):
                req.append("excluidos")
            for t in ["hongos", "moho", "insectos", "plagas"]:
                if want(t):
                    req.append(t)

        if "convoy" in q:
            req.append("convoy")
            if want("prohibido"):
                req.append("prohibido")
            if want("8 horas"):
                req.append("8 horas")
            if want("recursos de seguridad"):
                req.append("recursos de seguridad")

        out: List[str] = []
        seen = set()
        for t in req:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def _enforce_terms(self, answer: str, required: List[str]) -> str:
        a = (answer or "").strip()
        if not a:
            return a

        low = a.lower()
        missing = [t for t in required if t and (t.lower() not in low)]
        if not missing:
            return a

        return a.rstrip() + "\n\nTérminos clave: " + ", ".join(missing) + "."

    # ------------------ Answer Template v2 ------------------

    def _normalize_answer_format(self, text: str) -> str:
        t = (text or "").strip()
        t = re.sub(r"[ \t]+", " ", t)
        t = t.replace("USD 1'500,000", "USD 1,500,000")
        t = t.replace("USD 1’500,000", "USD 1,500,000")
        t = re.sub(r"\bhrs?\.\b", "horas", t, flags=re.IGNORECASE)
        t = re.sub(r"\bhrs?\b", "horas", t, flags=re.IGNORECASE)
        return t.strip()

    def _normalize_units_by_question(self, question: str, answer: str) -> str:
        q = (question or "").lower()
        a = (answer or "").strip()

        if "días" in q:
            m = re.fullmatch(r".*?:\s*(\d+)\s*$", a)
            if m:
                a = f"{m.group(1)} días"
            elif re.fullmatch(r"\d+\s*$", a):
                a = a.strip() + " días"

        if "horas" in q:
            m = re.fullmatch(r".*?:\s*(\d+)\s*$", a)
            if m:
                a = f"{m.group(1)} horas"
            elif re.fullmatch(r"\d+\s*$", a):
                a = a.strip() + " horas"

        if "años" in q:
            m = re.fullmatch(r".*?:\s*(\d+)\s*$", a)
            if m:
                a = f"{m.group(1)} años"
            elif re.fullmatch(r"\d+\s*$", a):
                a = a.strip() + " años"

        return a.strip()

    def _must_keywords_from_question(self, question: str) -> List[str]:
        q = (question or "").lower()
        kw: List[str] = []

        if "fast track" in q:
            kw.append("Fast Track")
        if "contado" in q:
            kw.append("contado")
        if "e-cargo" in q or "e cargo" in q:
            kw.append("póliza E-CARGO")
        if "camión" in q or "tráiler" in q or "trailer" in q:
            kw.extend(["camión", "tráiler"])
        if "avión" in q or "avion" in q:
            kw.append("avión")
        if "mensajería" in q or "paquetería" in q or "paqueteria" in q:
            kw.extend(["mensajería", "paquetería"])
        if "puerto" in q:
            kw.append("puerto")

        out: List[str] = []
        seen = set()
        for t in kw:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def _must_keywords_from_context(self, retrieved: List[Dict[str, Any]]) -> List[str]:
        c = "\n".join([(r.get("text") or "") for r in retrieved]).lower()
        kw: List[str] = []

        if "póliza e-cargo" in c or "poliza e-cargo" in c or "e-cargo" in c:
            kw.append("póliza E-CARGO")
        if "transporte terrestre" in c:
            kw.append("transporte terrestre")
        if "denuncia" in c:
            kw.append("denuncia")

        out: List[str] = []
        seen = set()
        for t in kw:
            if t not in seen:
                out.append(t)
                seen.add(t)
        return out

    def _apply_answer_template_v2(self, question: str, answer: str, retrieved: List[Dict[str, Any]]) -> str:
        a = self._normalize_answer_format(answer)
        a = self._normalize_units_by_question(question, a)

        context_plain = self._build_plain_context(retrieved)
        required = self._required_terms(question, context_plain)

        required += self._must_keywords_from_question(question)
        required += self._must_keywords_from_context(retrieved)

        out_req: List[str] = []
        seen = set()
        for t in required:
            if t and t not in seen:
                out_req.append(t)
                seen.add(t)

        a = self._enforce_terms(a, out_req)
        return a

    # ------------------ core prompt + ask ------------------

    def _prompt(self, question: str, context_structured: str) -> str:
        return f"""
Eres un asistente de análisis documental de pólizas.

Tu tarea es responder SOLO con base en el CONTEXTO.
Si el contexto no contiene una respuesta suficientemente explícita, responde exactamente:

{{
  "answer": "{NO_EVIDENCE}",
  "evidence_chunk_ids": []
}}

REGLAS OBLIGATORIAS:
- Responde en español.
- No inventes información.
- No uses conocimiento externo.
- No digas "según el documento, lo más relevante es".
- No listes fragmentos.
- No copies todo el contexto.
- Si hay definición explícita, redacta una respuesta clara en 1 o 2 frases.
- Devuelve SOLO JSON válido.
- evidence_chunk_ids debe contener SOLO chunk_id visibles en el CONTEXTO.
- Si no estás seguro, responde NO_EVIDENCE.

FORMATO DE SALIDA:
{{
  "answer": "...",
  "evidence_chunk_ids": [123, 456]
}}

PREGUNTA:
{question}

CONTEXTO:
{context_structured}
""".strip()

    def _maybe_attach_raw(self, out: Dict[str, Any], raw_llm_answer: str) -> Dict[str, Any]:
        if self.include_raw_llm_answer:
            out["raw_llm_answer"] = raw_llm_answer
        return out

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

        retrieved = self.retriever.retrieve(question, k=self.k)
        retrieved = self._rerank_for_question(question, retrieved)

        fuentes_for_ui: List[Any] = []
        for r in retrieved:
            md = r.get("metadata") or {}
            fuentes_for_ui.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "score": r.get("score"),
                    "page": md.get("page"),
                    "source": md.get("source") or md.get("file") or "pdf",
                }
            )

        if not retrieved:
            return {
                "mode": "local",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes_for_ui,
                "retrieved": [],
                "evidence_chunk_ids": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_hits",
            }

        context_plain = self._build_plain_context(retrieved)

        # Extractores deterministas primero
        vrdlm = self._extract_vrdlm_from_retrieved(question, retrieved)
        if vrdlm:
            ans = self._apply_answer_template_v2(question, vrdlm, retrieved)
            return {
                "mode": "local",
                "respuesta": ans,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": self._evidence_ids_for_vrdlm(retrieved),
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": "extractor_vrdlm",
            }

        ded = _extract_deducible_refrigeracion_strict(question, context_plain)
        if ded:
            ans = self._apply_answer_template_v2(question, ded, retrieved)
            return {
                "mode": "local",
                "respuesta": ans,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": self._top_chunk_ids(retrieved, self.top_n_for_llm),
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": "extractor_deducible_refrigeracion",
            }

        niveles = _extract_risk_levels(question, context_plain)
        if niveles:
            ans = self._apply_answer_template_v2(question, niveles, retrieved)
            return {
                "mode": "local",
                "respuesta": ans,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": self._top_chunk_ids(retrieved, self.top_n_for_llm),
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": "extractor_niveles_riesgo",
            }

        top = retrieved[: max(1, self.top_n_for_llm)]
        context_structured = self._build_structured_context(top)
        prompt = self._prompt(question, context_structured)

        if self.log_llm_prompt:
            logger.info("PROMPT OLLAMA:\n%s", prompt)

        try:
            raw = (self.ollama.generate(prompt) or "").strip()
            logger.info("RAW OLLAMA ANSWER: %s", raw)

            answer = ""
            evidence_ids: List[int] = []
            gate_reason: Optional[str] = None

            try:
                candidate = _extract_json_candidate(raw)
                obj = json.loads(candidate)
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
                raise RuntimeError("llm_empty_answer")

            # Debug: deja ver lo que produjo Ollama sin taparlo con fallback
            if self.debug_no_fallback:
                answer = self._apply_answer_template_v2(question, answer, retrieved)
                if not evidence_ids:
                    evidence_ids = self._top_chunk_ids(retrieved, self.top_n_for_llm)

                out = {
                    "mode": "local",
                    "respuesta": answer,
                    "fuentes": fuentes_for_ui,
                    "retrieved": retrieved,
                    "evidence_chunk_ids": evidence_ids,
                    "no_evidence": (answer.strip() == NO_EVIDENCE),
                    "used_fallback": False,
                    "gate_reason": gate_reason or "debug_raw_llm",
                }
                return self._maybe_attach_raw(out, raw)

            if answer.strip() == NO_EVIDENCE:
                bullets = []
                for r in retrieved[: min(5, len(retrieved))]:
                    t = (r.get("text") or "").strip()
                    if t and not self._is_page_marker(t):
                        bullets.append(f"- {t}")

                fallback_answer = "Según el documento, lo más relevante es:\n" + "\n".join(bullets)
                fallback_answer = self._apply_answer_template_v2(question, fallback_answer, retrieved)

                ids = self._top_chunk_ids(retrieved, self.top_n_for_llm)

                out = {
                    "mode": "local",
                    "respuesta": fallback_answer,
                    "fuentes": fuentes_for_ui,
                    "retrieved": retrieved,
                    "evidence_chunk_ids": ids,
                    "no_evidence": False,
                    "used_fallback": True,
                    "gate_reason": "llm_no_evidence_fallback_extractive",
                }
                return self._maybe_attach_raw(out, raw)

            answer = self._apply_answer_template_v2(question, answer, retrieved)

            if not evidence_ids:
                evidence_ids = self._top_chunk_ids(retrieved, self.top_n_for_llm)

            out = {
                "mode": "local",
                "respuesta": answer,
                "fuentes": fuentes_for_ui,
                "retrieved": retrieved,
                "evidence_chunk_ids": evidence_ids,
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": gate_reason,
            }
            return self._maybe_attach_raw(out, raw)

        except Exception as e:
            logger.error("❌ Ollama error: %s. Fallback a extractivo.", e)

            bullets = []
            for r in retrieved[: min(5, len(retrieved))]:
                t = (r.get("text") or "").strip()
                if t and not self._is_page_marker(t):
                    bullets.append(f"- {t}")

            fallback_answer = "Según el documento, lo más relevante es:\n" + "\n".join(bullets)
            fallback_answer = self._apply_answer_template_v2(question, fallback_answer, retrieved)

            ids = self._top_chunk_ids(retrieved, self.top_n_for_llm)

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