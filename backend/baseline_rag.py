from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from backend.baseline_store import BaselineStore

logger = logging.getLogger(__name__)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."
BASELINE_VERSION = "robust_v4.4_2026-03-03"

# ---------------------------
# Regex / Patterns
# ---------------------------

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{6,}\d)")

# “póliza: ABC-123…” (si existe explícito)
POLICY_RE = re.compile(
    r"(p[oó]liza|no\.?\s*de\s*p[oó]liza|n[uú]mero\s*de\s*p[oó]liza)\s*[:#]?\s*([A-Z0-9-]{4,})",
    re.IGNORECASE,
)

# token candidato a “ID de póliza” (más permisivo, pero SOLO se usa cuando el query lo exige)
POLICY_ID_RE = re.compile(r"\b[A-Z0-9][A-Z0-9-]{5,}\b")

DAYS_RE = re.compile(r"\b(\d{1,3})\s*(d[ií]as|d[ií]a)\b", re.IGNORECASE)
TIME_RE = re.compile(
    r"\b(\d{1,3})\s*(d[ií]as|d[ií]a|semanas|semana|meses|mes|horas|hora)\b",
    re.IGNORECASE,
)

PERCENT_RE = re.compile(r"\b(\d{1,3})\s*%\b")
POR_CIENTO_RE = re.compile(r"\b(\d{1,3})\s*por\s*ciento\b", re.IGNORECASE)

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

def _retrieved_texts(retrieved: List[Dict[str, Any]]) -> List[str]:
    return [(r.get("text") or "") for r in (retrieved or [])]

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

def _is_hard_vague(q: str) -> bool:
    """
    Vagas “duras” que en Fase 1 baseline-first deben abstener
    salvo que exista una sección claramente útil en el contexto.
    """
    ql = (q or "").strip().lower()
    hard = {
        "¿cómo funciona el seguro?",
        "¿como funciona el seguro?",
        "¿cómo funciona el seguro",
        "¿como funciona el seguro",
        "¿qué pasa si hay un problema con mi carga?",
        "¿que pasa si hay un problema con mi carga?",
        "¿qué pasa si hay un problema con mi carga",
        "¿que pasa si hay un problema con mi carga",
        "¿cuáles son las condiciones?",
        "¿cuales son las condiciones?",
        "¿cuáles son las condiciones",
        "¿cuales son las condiciones",
    }
    return ql in hard

def _is_boilerplate(text: str) -> bool:
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

# Anclas: ayudan a NO abstener por boilerplate/vague (pero no son suficientes para “hard vague”)
ANCHORS = [
    "prima", "factura", "pago", "mensual", "mínima", "minima", "plazo",
    "límite", "limite", "máximo", "maximo", "usd", "mxn", "$", "%",
    "cobertura", "exclus", "deducible", "suma asegurada", "vigencia",
    "responsabilidad", "indemn", "reclam", "siniestro", "aviso", "procedimiento",
    "documentos", "obligaciones",
    "requisitos", "obligatorios", "escolta", "monitoreo", "riesgo", "vrdlm",
    "cláusula", "clausula", "siniestralidad",
    "uber", "cabify", "hongos", "plagas", "buque", "años", "antigüedad", "antiguedad",
    "tecnología", "tecnologia", "5g",
]

def _has_anchor(ctx: str) -> bool:
    cl = (ctx or "").lower()
    return any(a in cl for a in ANCHORS)

def _has_useful_section(ctx: str) -> bool:
    """
    Señales de que realmente hay una sección accionable (no solo una lista o intro):
    - Procedimiento / aviso / reclamación
    - Cobertura / exclusiones / deducible / suma asegurada
    """
    cl = (ctx or "").lower()
    section_terms = [
        "procedimiento", "en caso de siniestro", "aviso", "reclamación", "reclamacion",
        "cobertura", "exclusiones", "exclusión", "exclusion", "deducible",
        "suma asegurada", "vigencia", "obligaciones",
    ]
    return any(t in cl for t in section_terms)

# ---------------------------
# Question Detection Gates
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

def _needs_reporting_days_for_damages(q: str) -> bool:
    ql = (q or "").lower()
    return _needs_reporting_days(q) and ("daños" in ql or "danos" in ql or "no robo" in ql or "no asalto" in ql)

def _needs_time_to_pay(q: str) -> bool:
    ql = (q or "").lower()
    return ("cuánto tiempo" in ql or "cuanto tiempo" in ql or "plazo" in ql) and ("pagar" in ql or "pago" in ql) and ("indemn" in ql)

def _needs_specific_ship(q: str) -> bool:
    ql = (q or "").lower()
    return "buque" in ql and ("libertador" in ql)

def _needs_drone_military_clause(q: str) -> bool:
    ql = (q or "").lower()
    return ("drone" in ql or "drones" in ql) and "militar" in ql

def _needs_email(q: str) -> bool:
    ql = (q or "").lower()
    return ("correo" in ql or "email" in ql or "e-mail" in ql) and ("contacto" in ql or "report" in ql or "siniestro" in ql or "chubb" in ql)

def _needs_phone(q: str) -> bool:
    ql = (q or "").lower()
    return ("teléfono" in ql or "telefono" in ql or "celular" in ql or "línea" in ql or "linea" in ql) and ("ajust" in ql or "contacto" in ql or "siniestro" in ql)

def _needs_percent(q: str) -> bool:
    ql = (q or "").lower()
    return ("porcentaje" in ql or "%" in ql or "porcentual" in ql) and (
        "cuota" in ql or "aplic" in ql or "valor asegurado" in ql or "siniestralidad" in ql
        or "proporción" in ql or "proporcion" in ql
    )

def _needs_percent_proportion_indemnizable(q: str) -> bool:
    ql = (q or "").lower()
    return ("porcentaje" in ql or "%" in ql) and ("proporción indemnizable" in ql or "proporcion indemnizable" in ql)

def _needs_5g_endorsement(q: str) -> bool:
    ql = (q or "").lower()
    return "5g" in ql

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
# Evidence Checks
# ---------------------------

def _evidence_has_policy_number(ctx: str) -> bool:
    # forma explícita (póliza: XXXX)
    return bool(POLICY_RE.search(ctx or ""))

def _evidence_has_policy_id_anywhere(retrieved_texts: List[str]) -> bool:
    """
    Para “número de póliza específico asignado a este contrato”:
    exigimos ver un ID con pinta de póliza en algún chunk.
    (No basta con decir “póliza” sin ID).
    """
    for t in retrieved_texts:
        tl = (t or "").lower()
        if "póliza" in tl or "poliza" in tl:
            # si el mismo chunk menciona póliza, buscamos tokens tipo ID
            if POLICY_RE.search(t) or POLICY_ID_RE.search(t):
                # filtrar tokens demasiado genéricos tipo "USD", "MXN", etc.
                # (si aparece "USD" será capturado por POLICY_ID_RE, lo evitamos)
                for m in POLICY_ID_RE.finditer(t):
                    tok = m.group(0)
                    if tok.upper() in {"USD", "MXN"}:
                        continue
                    # evitar tokens solo numéricos cortos
                    digits = re.sub(r"\D", "", tok)
                    if len(digits) >= 6 or ("-" in tok and len(tok) >= 7):
                        return True
    return False

def _evidence_has_reporting_days(ctx: str) -> bool:
    if not ctx:
        return False
    cl = ctx.lower()
    if ("report" not in cl) and ("avis" not in cl):
        return False
    return bool(DAYS_RE.search(ctx))

def _evidence_has_reporting_days_for_damages(ctx: str) -> bool:
    """
    Para “daños (no robo)” exigimos:
    - que haya días
    - y que el contexto mencione daños / no robo / no asalto (o equivalente)
    """
    if not _evidence_has_reporting_days(ctx):
        return False
    cl = (ctx or "").lower()
    return ("daños" in cl or "danos" in cl or "no robo" in cl or "no asalto" in cl or "daño" in cl or "dano" in cl)

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
    if PERCENT_RE.search(ctx):
        return True
    return bool(POR_CIENTO_RE.search(ctx))

def _evidence_has_percent_near_proportion(retrieved_texts: List[str]) -> bool:
    """
    Para “proporción indemnizable”, exigir que el % aparezca
    en el MISMO chunk donde aparece 'propor' o 'indemniz'.
    Evita falsos positivos cuando otro chunk tiene '%'.
    """
    for t in retrieved_texts:
        tl = (t or "").lower()
        if ("propor" in tl or "indemniz" in tl):
            if PERCENT_RE.search(t) or POR_CIENTO_RE.search(t):
                return True
    return False

def _evidence_has_time(ctx: str) -> bool:
    if not ctx:
        return False
    return bool(TIME_RE.search(ctx))

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

        ctx = _join_text(retrieved)
        ctx_low = ctx.lower()
        texts = _retrieved_texts(retrieved)

        # ------------------------
        # Required-pattern gates (exact info)
        # ------------------------

        # Histórico externo: por definición no está en el PDF
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

        # Email requerido
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

        # Teléfono requerido
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

        # Porcentaje general requerido
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

        # Porcentaje “proporción indemnizable” (más estricto: % debe estar cerca)
        if _needs_percent_proportion_indemnizable(question) and not _evidence_has_percent_near_proportion(texts):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(percent_near_proportion)",
                "baseline_version": BASELINE_VERSION,
            }

        # Número de póliza específico: exige ID real (no basta mencionar póliza)
        if _needs_policy_number(question):
            # primero: forma explícita
            if not _evidence_has_policy_number(ctx):
                # fallback: buscar “ID con pinta” en chunks donde mencionan póliza
                if not _evidence_has_policy_id_anywhere(texts):
                    return {
                        "mode": "baseline",
                        "respuesta": NO_EVIDENCE,
                        "fuentes": [],
                        "retrieved": retrieved,
                        "no_evidence": True,
                        "used_fallback": False,
                        "gate_reason": "missing_required_pattern(policy_id)",
                        "baseline_version": BASELINE_VERSION,
                    }

        # Reporte de siniestro: días
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

        # Reporte de siniestro por daños (no robo): exige contexto de daños/no robo + días
        if _needs_reporting_days_for_damages(question) and not _evidence_has_reporting_days_for_damages(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(reporting_days_damages)",
                "baseline_version": BASELINE_VERSION,
            }

        # Tiempo de pago de indemnización: exige número + unidad temporal
        if _needs_time_to_pay(question) and not _evidence_has_time(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_pattern(time_to_pay)",
                "baseline_version": BASELINE_VERSION,
            }

        # País fuera del alcance: si preguntan por un país y no aparece en evidencia => abstener
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

        # Endoso/tema específico 5G: si no aparece “5g” en evidencia => abstener
        if _needs_5g_endorsement(question) and ("5g" not in ctx_low):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "missing_required_topic(5g)",
                "baseline_version": BASELINE_VERSION,
            }

        # ------------------------
        # Conservative gates (boilerplate/vague)
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

        # Vague “hard”: SOLO responder si hay una sección útil clara
        if _is_hard_vague(question) and not _has_useful_section(ctx):
            return {
                "mode": "baseline",
                "respuesta": NO_EVIDENCE,
                "fuentes": [],
                "retrieved": retrieved,
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "hard_vague_no_section",
                "baseline_version": BASELINE_VERSION,
            }

        # Vague normal: SOLO abstener si NO hay ancla
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
        # Existing Abstention Gates (topics/entities)
        # ------------------------

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