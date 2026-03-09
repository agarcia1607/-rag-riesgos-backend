from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from backend.baseline_rag import BaselineRAG
from backend.local_rag import LocalRAG

logger = logging.getLogger(__name__)

# Mensaje estándar esperado por los tests
NO_EVIDENCE_STD = "No encontré fragmentos relevantes para esa consulta en el documento."


class ChatbotRiesgos:
    """
    Wrapper que enruta consultas a baseline/local y normaliza la salida
    para mantener un contrato estable hacia FastAPI/tests/evaluación.
    """

    def __init__(
        self,
        pdf_path: Optional[str] = None,
        mode: Optional[str] = None,
        debug: bool = False,
        k: int = 5,
    ):
        self.pdf_path = pdf_path or os.getenv("PDF_PATH", "data/Doc chatbot.pdf")
        self.mode = (mode or os.getenv("RAG_MODE", "local")).lower()
        self.debug = debug
        self.k = int(os.getenv("BM25_K", str(k)))

        self.baseline = BaselineRAG(
            pdf_path=self.pdf_path,
            debug=self.debug,
            k=self.k,
        )
        self.local = LocalRAG(
            pdf_path=self.pdf_path,
            k=self.k,
        )

        self.baseline_version = getattr(self.baseline, "BASELINE_VERSION", None)
        if self.baseline_version is None:
            self.baseline_version = getattr(
                __import__("backend.baseline_rag", fromlist=["BASELINE_VERSION"]),
                "BASELINE_VERSION",
                None,
            )

        logger.info(
            "✅ ChatbotRiesgos listo | mode=%s | pdf=%s | k=%s",
            self.mode,
            self.pdf_path,
            self.k,
        )

    def _resolve_mode(self, mode: Optional[str]) -> str:
        requested = (mode or self.mode or "local").lower().strip()
        if requested in {"baseline", "local"}:
            return requested
        return "local"

    def _normalize_out(self, out: Dict[str, Any], *, requested_mode: str) -> Dict[str, Any]:
        """
        Asegura contrato estable:
        - baseline_version siempre presente (si se conoce)
        - no_evidence siempre bool
        - retrieved siempre lista
        - used_fallback siempre bool
        - gate_reason siempre str|None
        - requested_mode se reporta para debugging/evaluación
        - unifica mensajes legacy de no-evidence al contrato estándar del proyecto
        """
        if not isinstance(out, dict):
            out = {"respuesta": str(out)}

        out.setdefault("requested_mode", requested_mode)
        out.setdefault("baseline_version", self.baseline_version)

        out.setdefault("fuentes", [])
        out.setdefault("retrieved", [])
        out.setdefault("used_fallback", False)
        out.setdefault("gate_reason", None)

        legacy_no_evidence_messages = {
            "No se encontró evidencia suficiente en los documentos.",
            "No encontré fragmentos relevantes para esa consulta en el documento.",
            "",
            None,
        }

        respuesta = out.get("respuesta", "")

        inferred_no_evidence = (
            respuesta in legacy_no_evidence_messages
            or (out.get("fuentes") == [] and out.get("retrieved") == [])
        )

        no_evidence = out.get("no_evidence", None)
        if no_evidence is None:
            out["no_evidence"] = bool(inferred_no_evidence)
        else:
            out["no_evidence"] = bool(no_evidence)

        if out["no_evidence"]:
            out["respuesta"] = NO_EVIDENCE_STD
            out["fuentes"] = out.get("fuentes") or []

        if not out.get("respuesta"):
            out["respuesta"] = NO_EVIDENCE_STD
            out["no_evidence"] = True
            out["gate_reason"] = out.get("gate_reason") or "empty_answer_normalized"

        return out

    def consultar(
        self,
        pregunta: str,
        mostrar_fuentes: bool = True,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        effective_mode = self._resolve_mode(mode)

        logger.info("🔍 Consulta modo=%s: %s", effective_mode, pregunta)

        if effective_mode == "baseline":
            out = self.baseline.ask(pregunta)
        else:
            out = self.local.ask(pregunta)

        out = self._normalize_out(out, requested_mode=effective_mode)

        if not mostrar_fuentes:
            out["fuentes"] = []

        return out