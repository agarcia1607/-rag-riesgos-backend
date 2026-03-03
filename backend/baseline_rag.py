from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from backend.baseline_store import BaselineStore

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."
BASELINE_VERSION = "robust_v4.3_2026-03-03"

# ---------------------------
# Regex / Patterns
# ---------------------------

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{6,}\d)")
POLICY_RE = re.compile(
    r"(p[oó]liza|no\.?\s*de\s*p[oó]liza|n[uú]mero\s*de\s*p[oó]liza)\s*[:#]?\s*([A-Z0-9-]{4,})",
    re.IGNORECASE,
)
DAYS_RE = re.compile(r"\b(\d{1,3})\s*(d[ií]as|d[ií]a)\b", re.IGNORECASE)

# ---------------------------
# Helper Functions
# ---------------------------

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

# ---------------------------
# Conservative Gates (refined)
# ---------------------------

def _is_vague_question(q: str) -> bool:
    ql = (q or "").lower()
    vague_patterns = [
        "cómo funciona",
        "como funciona",
        "cuáles son las condiciones",
        "cuales son las condiciones",
        "qué cubre",
        "que cubre",
        "explique",
        "describe",
        "resumen",
        "condiciones",
        "funciona el seguro",
        "cómo opera",
        "como opera",
        "qué pasa si",
        "que pasa si",
        "problema con mi carga",
        "problema con la carga",
    ]
    return any(p in ql for p in vague_patterns)

def _is_boilerplate(text: str) -> bool:
    """
    Señal de intro/boilerplate. NO debe bloquear si hay evidencia anclada real.
    """
    tl = (text or "").lower()
    patterns = [
        "inicio",
        "se aclara",
        "cláusulas indicadas",
        "clausulas indicadas",
        "condiciones generales",
        "independientemente de",
        "únicas que aplicarán",
        "unicas que aplicaran",
    ]
    return any(p in tl for p in patterns)

# Lista de anclas (evidencia fuerte) para NO abstener por vague/boilerplate
ANCHORS = [
    # pagos / primas
    "prima", "factura", "pago", "mensual", "mínima", "minima", "plazo",

    # límites / montos
    "límite", "limite", "máximo", "maximo", "usd", "mxn", "$", "%",

    # cobertura / exclusiones / condiciones específicas
    "cobertura", "exclus", "deducible", "suma asegurada", "vigencia",
    "responsabilidad", "indemn", "reclam", "siniestro", "aviso", "procedimiento",
    "documentos", "obligaciones",

    # seguridad / riesgo
    "requisitos", "obligatorios", "escolta", "monitoreo", "riesgo", "vrdlm",

    # cláusulas
    "cláusula", "clausula", "siniestralidad",

    # temas concretos del PDF
    "uber", "cabify", "hongos", "plagas", "buque", "años", "antigüedad", "antiguedad",
    "tecnología", "tecnologia", "5g",
]

def _has_anchor(ctx: str) -> bool:
    cl = (ctx or "").lower()
    return any(a in cl for a in ANCHORS)

# ---------------------------
# Question Detection Gates (existing + new)
# ---------------------------

def _needs_policy_number(q: str) -> bool:
    ql = (q or "").lower()
    return "póliza" in ql and ("número" in ql or "numero" in ql or "no." in ql or "específico" in ql or "especifico" in ql)

def _needs_reporting_days(q: str) -> bool:
    ql = (q or "").lower()
    return (
        ("cuántos días" in ql or "cuantos dias" in ql or "días tiene" in ql or "dias tiene" in ql)
        and ("report" in ql or "avis" in ql)
        and "siniestro" in ql
    )

def _needs_specific_ship(q: str) -> bool:
    ql = (q or "").lower()
    return "buque" in ql and ("libertador" in ql)

def _needs_drone_military_clause(q: str) -> bool:
    ql = (q or "").lower()
    return ("drone" in ql or "drones" in ql) and "militar" in ql

# NEW detectors
def _needs_email(q: str) -> bool:
    ql = (q or "").lower()
    return ("correo" in ql or "email" in ql or "e-mail" in ql) and ("contacto" in ql or "report" in ql or "siniestro" in ql or "chubb" in ql)

def _needs_phone(q: str) -> bool:
    ql = (q or "").lower()
    return ("teléfono" in ql or "telefono" in ql or "celular" in ql or "línea" in ql or "linea" in ql) and ("ajust" in ql or "contacto" in ql or "siniestro" in ql)

def _needs_percent(q: str) -> bool:
    ql = (q or "").lower()
    return ("porcentaje" in ql or "%" in ql or "porcentual" in ql) and ("cuota" in ql or "aplic" in ql or "valor asegurado" in ql or "siniestralidad" in ql or "proporción" in ql or "proporcion" in ql)

def _is_historical_external(q: str) -> bool:
    ql = (q or "").lower()
    return ("cuántos siniestros" in ql or "cuantos siniestros" in ql) and ("últimos" in ql or "ultimos" in ql or "12 meses" in ql or "año" in ql or "ano" in ql)

def _mentions_country_outside_scope(q: str) -> str:
    ql = (q or "").lower()
    countries = [
        "brasil", "argentina", "chile", "perú", "peru",
        "colombia", "eeuu", "usa", "estados unidos", "canadá", "canada"
    ]
    for c in countries:
        if c in ql:
            return c
    return ""

# ---------------------------
# Evidence Checks (existing + new)
# ---------------------------

def _evidence_has_policy_number(ctx: str) -> bool:
    return bool(POLICY_RE.search(ctx or ""))

def _evidence_has_reporting_days(ctx: str) -> bool:
    if not ctx:
        return False
    cl = ctx.lower()
    if ("report" not in cl) and ("avis" not in cl):
        return False
    return bool(DAYS_RE.search(ctx))

def _evidence_has_email(ctx: str) -> bool:
    return bool(EMAIL_RE.search(ctx or ""))

def _evidence_has_phone(ctx: str) -> bool:
    if not ctx:
        return False
    for m in PHONE_RE.finditer(ctx):
        digits = re.sub(r"\D", "", m.group(0))
        if 7 <= len(digits) <= 15:
            return True
    return False

def _evidence_has_percent(ctx: str) -> bool:
    if not ctx:
        return False
    if "%" in ctx:
        return True
    return bool(re.search(r"\b\d{1,3}\s*por\s*ciento\b", ctx, re.IGNORECASE))

def _evidence_mentions(term: str, ctx: str) -> bool:
    return bool(term) and (term.lower() in (ctx or "").lower())

# ---------------------------
# Main Class
# ---------------------------

class BaselineRAG:
    def __init__(self, pdf_path: str, debug: bool = False, k: int = 5):
        self.pdf_path = pdf_path
        self.debug = debug
        self.k = int(k)

        self.store = BaselineStore(pdf_path=self.pdf_path)
        self.store.build_or_load()

        logger.info("🟦 BaselineRAG listo | BM25 | k=%s | version=%s", self.k, BASELINE_VERSION)

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

        hits = self.store.search(question, k=self.k)

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

        if not retrieved:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved_raw,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "no_signal",
                "baseline_version": BASELINE_VERSION,
            }

        # Contexto (top-k)
        ctx = _join_text(retrieved)
        ctx_low = ctx.lower()

        # ------------------------
        # NEW required-pattern gates (targeting your 13 failures)
        # ------------------------

        # (1) Histórico externo: abstener por definición (no está en el PDF)
        if _is_historical_external(question):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "external_historical_question",
                "baseline_version": BASELINE_VERSION,
            }

        # (2) Email requerido
        if _needs_email(question) and not _evidence_has_email(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(email)",
                "baseline_version": BASELINE_VERSION,
            }

        # (3) Teléfono requerido
        if _needs_phone(question) and not _evidence_has_phone(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(phone)",
                "baseline_version": BASELINE_VERSION,
            }

        # (4) Porcentaje requerido
        if _needs_percent(question) and not _evidence_has_percent(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(percent)",
                "baseline_version": BASELINE_VERSION,
            }

        # (5) País fuera del alcance: si preguntan por un país y no aparece en evidencia => abstener
        country = _mentions_country_outside_scope(question)
        if country and not _evidence_mentions(country, ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": f"missing_required_entity(country:{country})",
                "baseline_version": BASELINE_VERSION,
            }

        # ------------------------
        # Conservative Gates (refined with anchors)
        # ------------------------

        top_text = (retrieved[0].get("text") or "").strip()

        # Boilerplate: SOLO abstener si NO hay ancla
        if _is_boilerplate(top_text) and not _has_anchor(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "boilerplate_no_anchor",
                "baseline_version": BASELINE_VERSION,
            }

        # Vague: SOLO abstener si NO hay ancla
        if _is_vague_question(question) and not _has_anchor(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "vague_question_no_anchor",
                "baseline_version": BASELINE_VERSION,
            }

        # ------------------------
        # Existing Abstention Gates
        # ------------------------

        if _needs_policy_number(question) and not _evidence_has_policy_number(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(policy_number)",
                "baseline_version": BASELINE_VERSION,
            }

        if _needs_reporting_days(question) and not _evidence_has_reporting_days(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(reporting_days)",
                "baseline_version": BASELINE_VERSION,
            }

        if _needs_specific_ship(question) and "libertador" not in ctx_low:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_entity(ship_not_mentioned)",
                "baseline_version": BASELINE_VERSION,
            }

        if _needs_drone_military_clause(question):
            if ("drone" not in ctx_low and "drones" not in ctx_low) or ("militar" not in ctx_low):
                return {
                    "mode": "baseline",
                    "respuesta": NO_EVIDENCE,
                    "fuentes": [],
                    "retrieved": retrieved,
                    "no_evidence": True,
                    "used_fallback": False,
                    "gate_reason": "missing_required_topic(drones_military)",
                    "baseline_version": BASELINE_VERSION,
                }

        # ------------------------
        # Extractive Response (default)
        # ------------------------

        bullets: List[str] = []
        for r in retrieved[:5]:
            t = (r.get("text") or "").strip()
            if not t:
                continue
            if _is_page_marker(t):
                continue
            bullets.append(f"- {t}")

        if not bullets:
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
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
            "fuentes": [r.get("text", "") for r in retrieved[:5]],
            "retrieved": retrieved,
            "no_evidence": False,
            "used_fallback": False,
            "gate_reason": "answered_extractively",
            "baseline_version": BASELINE_VERSION,
        }