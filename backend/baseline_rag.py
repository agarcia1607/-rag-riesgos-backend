from __future__ import annotations

from typing import Dict, Any, List, Tuple
import logging
import re

from backend.baseline_store import BaselineStore

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."


# ---------------------------
# Helpers / Gates
# ---------------------------

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
# Teléfono “tolerante”: detecta secuencias razonables de dígitos (con espacios/guiones/paréntesis)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{6,}\d)")
# Póliza: algo tipo “póliza: ABC123…” o “No. de póliza …”
POLICY_RE = re.compile(
    r"(p[oó]liza|no\.?\s*de\s*p[oó]liza|n[uú]mero\s*de\s*p[oó]liza)\s*[:#]?\s*([A-Z0-9-]{4,})",
    re.IGNORECASE,
)


def _has_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term in t for term in terms)


def _is_page_marker(s: str) -> bool:
    return bool(re.fullmatch(r"\[PAGE\s+\d+\]", (s or "").strip(), re.IGNORECASE))


def _filter_noise(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in retrieved or []:
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        if _is_page_marker(txt):
            continue
        # opcional: filtrar tabla de contenidos si te mete ruido (lo dejamos pasar por ahora)
        out.append(r)
    return out


def _join_text(retrieved: List[Dict[str, Any]]) -> str:
    return " ".join([(r.get("text") or "") for r in retrieved or []])


def _max_score(retrieved: List[Dict[str, Any]]) -> float:
    scores: List[float] = []
    for r in retrieved or []:
        try:
            scores.append(float(r.get("score", 0.0)))
        except Exception:
            pass
    return max(scores) if scores else 0.0


def _is_vague_question(q: str) -> bool:
    ql = (q or "").strip().lower()
    vague_patterns = [
        "¿cómo funciona",
        "como funciona",
        "¿cuáles son las condiciones",
        "cuales son las condiciones",
        "¿qué pasa si",
        "que pasa si",
        "tengo un problema con mi carga",
    ]
    if any(p in ql for p in vague_patterns):
        return True
    if len(ql) <= 18 and ("condiciones" in ql or "funciona" in ql):
        return True
    return False


def _needs_email(q: str) -> bool:
    ql = (q or "").lower()
    return ("correo" in ql) or ("email" in ql) or ("e-mail" in ql)


def _needs_phone(q: str) -> bool:
    ql = (q or "").lower()
    return ("tel" in ql) or ("teléfono" in ql) or ("telefono" in ql) or ("whatsapp" in ql)


def _needs_policy_number(q: str) -> bool:
    ql = (q or "").lower()
    return ("número de póliza" in ql) or ("numero de poliza" in ql) or ("no. de póliza" in ql) or ("no de póliza" in ql)


def _needs_percent_quota(q: str) -> bool:
    ql = (q or "").lower()
    return ("cuota base" in ql) and ("%" in ql or "porcent" in ql)


def _needs_historical_claims(q: str) -> bool:
    ql = (q or "").lower()
    return ("cuántos siniestros" in ql or "cuantos siniestros" in ql) and ("últimos" in ql or "ultimos" in ql)


def _needs_indemnity_payment_time(q: str) -> bool:
    ql = (q or "").lower()
    return ("tarda" in ql or "cuánto tiempo" in ql or "cuanto tiempo" in ql) and ("pagar" in ql or "pago" in ql) and ("indemn" in ql)


def _needs_country_coverage(q: str) -> bool:
    ql = (q or "").lower()
    return "cubre" in ql and ("brasil" in ql or "argentina" in ql or "chile" in ql or "perú" in ql or "peru" in ql)


def _needs_proporcion_percent(q: str) -> bool:
    ql = (q or "").lower()
    return ("proporción indemnizable" in ql or "proporcion indemnizable" in ql) and ("porcentaje" in ql or "%" in ql)


def _evidence_has_email(ctx: str) -> bool:
    return bool(EMAIL_RE.search(ctx or ""))


def _evidence_has_phone(ctx: str) -> bool:
    m = PHONE_RE.search(ctx or "")
    if not m:
        return False
    digits = re.sub(r"\D", "", m.group(1))
    return len(digits) >= 7


def _evidence_has_policy_number(ctx: str) -> bool:
    return bool(POLICY_RE.search(ctx or ""))


def _evidence_has_percent_near(ctx: str, keyword: str, window: int = 80) -> bool:
    if not ctx:
        return False
    low = ctx.lower()
    k = keyword.lower()
    i = low.find(k)
    if i == -1:
        return False
    start = max(0, i - window)
    end = min(len(ctx), i + window)
    span = ctx[start:end]
    return "%" in span or "porcent" in span.lower()


class BaselineRAG:
    """
    Baseline (extractivo, reproducible):
    - Retrieval: BM25 (BaselineStore)
    - Respuesta: extractiva / bullet points desde los chunks

    Gates (robustos):
    - empty_question
    - no_hits (real)
    - no_signal(all_noise)
    - no_signal(score<=0)
    - required_topic_missing(cyber)
    - external_entity_question_gate
    - missing_required_pattern(email/phone/policy/percent/etc.)
    - external_data_required(historical_claims)
    - missing_clause(payment_time)
    - missing_required_entity(country_not_mentioned)
    - missing_required_pattern(proporcion_percent)
    - vague_question
    """

    def __init__(self, pdf_path: str, debug: bool = False, k: int = 5):
        self.pdf_path = pdf_path
        self.debug = debug
        self.k = k

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

        retrieved_raw = self._hits_to_retrieved(hits)
        retrieved = _filter_noise(retrieved_raw)
        fuentes = [r.get("text", "") for r in retrieved]

        # ✅ Gate 0: todo era ruido ([PAGE x] o vacío)
        if not retrieved:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved_raw,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_signal(all_noise)",
            }

        # ✅ Gate 1: BM25 sin señal (scores 0 o casi 0)
        # Ajusta umbral si quieres: 0.0 / 0.05 / 0.1 según tu PDF.
        if _max_score(retrieved) <= 0.0:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_signal(score<=0)",
            }

        q = question.lower()
        ctx = _join_text(retrieved)
        ctx_low = ctx.lower()

        # ✅ Gate: preguntas vagas (en tu dataset, esto debería abstener)
        if _is_vague_question(question):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "vague_question",
            }

        # ✅ Gate: cyber
        cyber_q_terms = ["ciber", "cibern", "cyber", "ransom", "malware", "phishing", "ddos", "hack"]
        cyber_ctx_terms = cyber_q_terms + ["intrusión", "intrusion", "ataque", "ataques"]
        if _has_any(q, cyber_q_terms) and not _has_any(ctx_low, cyber_ctx_terms):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "required_topic_missing(cyber)",
            }

        # ✅ Gate: entidad externa (CEO, etc.)
        external_terms = ["ceo", "director general", "presidente", "owner", "propietario", "gerente general"]
        if _has_any(q, external_terms):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "external_entity_question_gate",
            }

        # ✅ Gate: patrones requeridos (email / teléfono / póliza / % cuota / % proporción)
        if _needs_email(question) and not _evidence_has_email(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(email)",
            }

        if _needs_phone(question) and not _evidence_has_phone(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(phone)",
            }

        if _needs_policy_number(question) and not _evidence_has_policy_number(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(policy_number)",
            }

        if _needs_percent_quota(question) and not _evidence_has_percent_near(ctx, "cuota"):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(percent_quota)",
            }

        if _needs_historical_claims(question):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "external_data_required(historical_claims)",
            }

        if _needs_indemnity_payment_time(question):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_clause(payment_time)",
            }

        if _needs_country_coverage(question):
            if "brasil" in q and "brasil" not in ctx_low:
                return {
                    "mode": "baseline",
                    "respuesta": NO_EVIDENCE,
                    "fuentes": fuentes,
                    "retrieved": retrieved,
                    "no_evidence": True,
                    "used_fallback": False,
                    "gate_reason": "missing_required_entity(country_not_mentioned)",
                }

        if _needs_proporcion_percent(question):
            if not _evidence_has_percent_near(ctx, "proporción") and not _evidence_has_percent_near(ctx, "proporcion"):
                return {
                    "mode": "baseline",
                    "respuesta": NO_EVIDENCE,
                    "fuentes": fuentes,
                    "retrieved": retrieved,
                    "no_evidence": True,
                    "used_fallback": False,
                    "gate_reason": "missing_required_pattern(proporcion_percent)",
                }

        # ------------------------
        # Extractivo simple
        # ------------------------
        bullets: List[str] = []
        for r in retrieved[: min(5, len(retrieved))]:
            t = (r.get("text") or "").strip()
            if not t:
                continue
            bullets.append(f"- {t}")

        if not bullets:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_signal(no_bullets)",
            }

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