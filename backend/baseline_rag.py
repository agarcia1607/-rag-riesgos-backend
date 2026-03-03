from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from backend.baseline_store import BaselineStore

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."
BASELINE_VERSION = "robust_v4.2_2026-03-03"

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
    ]
    return any(p in ql for p in vague_patterns)

def _is_boilerplate(text: str) -> bool:
    """
    OJO: esto NO debe disparar abstención si hay anclas de evidencia.
    Solo sirve como señal de "intro/boilerplate".
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
        # Nota: NO incluyas "según el documento" aquí; eso lo generas tú en la respuesta.
    ]
    return any(p in tl for p in patterns)

# Lista de anclas (evidencia fuerte) para NO abstener por vague/boilerplate
ANCHORS = [
    # pagos / primas
    "prima", "factura", "pago", "mensual", "mínima", "minima",

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

    # entidades/temas concretos del PDF
    "uber", "cabify", "hongos", "plagas", "buque", "años", "antigüedad", "antiguedad",
]

def _has_anchor(ctx: str) -> bool:
    cl = (ctx or "").lower()
    return any(a in cl for a in ANCHORS)

# ---------------------------
# Existing Question Detection Gates
# ---------------------------

def _needs_policy_number(q: str) -> bool:
    ql = (q or "").lower()
    return "póliza" in ql and ("número" in ql or "numero" in ql or "no." in ql)

def _needs_reporting_days(q: str) -> bool:
    ql = (q or "").lower()
    return (
        ("cuántos días" in ql or "cuantos dias" in ql)
        and ("report" in ql or "avis" in ql)
        and "siniestro" in ql
    )

def _needs_specific_ship(q: str) -> bool:
    ql = (q or "").lower()
    return "buque" in ql and ("libertador" in ql)

def _needs_drone_military_clause(q: str) -> bool:
    ql = (q or "").lower()
    return ("drone" in ql or "drones" in ql) and "militar" in ql

# ---------------------------
# Evidence Checks (existing)
# ---------------------------

def _evidence_has_policy_number(ctx: str) -> bool:
    return bool(POLICY_RE.search(ctx or ""))

def _evidence_has_reporting_days(ctx: str) -> bool:
    if not ctx:
        return False
    if ("report" not in ctx.lower()) and ("avis" not in ctx.lower()):
        return False
    return bool(DAYS_RE.search(ctx))

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

        logger.info("🟦 BaselineRAG listo | BM25 inicializado | k=%s | version=%s", self.k, BASELINE_VERSION)

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
        # Conservative Gates (refined with anchors)
        # ------------------------

        top_text = (retrieved[0].get("text") or "").strip()

        # Boilerplate: SOLO abstener si además NO hay ancla (si hay ancla, es evidencia útil)
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

        # Pregunta vaga: SOLO abstener si NO hay ancla en el contexto
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