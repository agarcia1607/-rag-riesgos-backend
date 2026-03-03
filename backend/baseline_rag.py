from __future__ import annotations

from typing import Dict, Any, List, Tuple
import logging
import re

from backend.baseline_store import BaselineStore

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."
BASELINE_VERSION = "robust_v3_debug_2026-03-03"
# ---------------------------
# Regex / Patterns
# ---------------------------

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)

# Teléfono tolerante: +57 300..., (55) 1234-5678, 01-800-..., etc.
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{6,}\d)")

# Póliza: “póliza: ABC123”, “No. de póliza ABC123”, “número de póliza XYZ-999”
POLICY_RE = re.compile(
    r"(p[oó]liza|no\.?\s*de\s*p[oó]liza|n[uú]mero\s*de\s*p[oó]liza)\s*[:#]?\s*([A-Z0-9-]{4,})",
    re.IGNORECASE,
)

# ---------------------------
# Helpers / Gates
# ---------------------------


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

    # Heurística: demasiado amplia / no accionable
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

    # muy corta y genérica
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
    return (
        ("número de póliza" in ql)
        or ("numero de poliza" in ql)
        or ("no. de póliza" in ql)
        or ("no de póliza" in ql)
        or ("nro de póliza" in ql)
    )


def _needs_percent_quota(q: str) -> bool:
    ql = (q or "").lower()
    return ("cuota base" in ql) and (("%" in ql) or ("porcent" in ql))


def _needs_historical_claims(q: str) -> bool:
    ql = (q or "").lower()
    return ("cuántos siniestros" in ql or "cuantos siniestros" in ql) and ("últimos" in ql or "ultimos" in ql)


def _needs_indemnity_payment_time(q: str) -> bool:
    ql = (q or "").lower()
    return (("tarda" in ql) or ("cuánto tiempo" in ql) or ("cuanto tiempo" in ql)) and ("pagar" in ql or "pago" in ql) and ("indemn" in ql)


def _needs_country_coverage(q: str) -> bool:
    ql = (q or "").lower()
    return ("cubre" in ql) and any(c in ql for c in ["brasil", "argentina", "chile", "perú", "peru", "colombia", "méxico", "mexico"])


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
    return ("%" in span) or ("porcent" in span.lower())


class BaselineRAG:
    """
    Baseline (extractivo, reproducible):
    - Retrieval: BM25 (BaselineStore)
    - Respuesta: extractiva / bullet points desde los chunks

    Gates (robustos):
    - empty_question
    - no_hits (real)
    - no_signal (BM25 sin señal: scores ~ 0 o todo ruido)
    - vague_question
    - required_topic_missing(cyber)
    - external_entity_question_gate
    - missing_required_pattern(email/phone/policy/percent/etc.)
    - external_data_required(historical_claims)
    - missing_clause(payment_time)
    - missing_required_entity(country_not_mentioned)
    - missing_required_pattern(proporcion_percent)
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
                "baseline_version": BASELINE_VERSION,
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
                "baseline_version": BASELINE_VERSION,
            }

        retrieved_raw = self._hits_to_retrieved(hits)
        retrieved = _filter_noise(retrieved_raw)
        fuentes = [r.get("text", "") for r in retrieved]

        # Gate: todo era ruido
        if not retrieved:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved_raw,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_signal(all_noise)",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: BM25 sin señal
        # OJO: rank_bm25 puede dar 0 para muchas queries. Si te queda muy agresivo, cambia a <= 0.05
        if _max_score(retrieved) <= 0.0:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_signal(score<=0)",
                "baseline_version": BASELINE_VERSION,
            }

        q = question.lower()
        ctx = _join_text(retrieved)
        ctx_low = ctx.lower()

        # Gate: pregunta vaga
        if _is_vague_question(question):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "vague_question",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: cyber
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
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: entidad externa
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
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: email
        if _needs_email(question) and not _evidence_has_email(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(email)",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: phone
        if _needs_phone(question) and not _evidence_has_phone(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(phone)",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: policy number
        if _needs_policy_number(question) and not _evidence_has_policy_number(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(policy_number)",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: cuota base porcentual
        if _needs_percent_quota(question) and not _evidence_has_percent_near(ctx, "cuota"):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(percent_quota)",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: siniestros últimos 12 meses (dato externo)
        if _needs_historical_claims(question):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "external_data_required(historical_claims)",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: tiempo de pago indemnización (cláusula usualmente no está)
        if _needs_indemnity_payment_time(question):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": fuentes,
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_clause(payment_time)",
                "baseline_version": BASELINE_VERSION,
            }

        # Gate: cobertura por país específico (si país no aparece, abstener)
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
                    "baseline_version": BASELINE_VERSION,
                }

        # Gate: porcentaje exacto de proporción indemnizable
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
                    "baseline_version": BASELINE_VERSION,
                }

        # ------------------------
        # Extractivo simple (no inventa)
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
                "baseline_version": BASELINE_VERSION,
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
            "baseline_version": BASELINE_VERSION,
        }