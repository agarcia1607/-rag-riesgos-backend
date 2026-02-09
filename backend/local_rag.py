from __future__ import annotations

from typing import Dict, Any, List, Tuple
import re

from backend.baseline_rag import BaselineRAG
from backend.ollama_client import OllamaClient


NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."


def _normalize(text: str) -> str:
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _looks_like_meta_or_disclaimer(answer: str) -> bool:
    a = (answer or "").lower()
    bad_markers = [
        "como modelo", "soy un modelo", "modelo de lenguaje", "no tengo acceso",
        "no puedo acceder", "no tengo capacidad", "no tengo información",
        "no tengo el contexto", "no puedo ver", "como asistente", "ia creada por",
        "alibaba", "openai", "no estoy seguro"
    ]
    return any(m in a for m in bad_markers)


def _overlap_ratio(answer: str, context: str) -> float:
    """
    Heurística rápida: qué tanto vocabulario del answer aparece en el contexto.
    No es perfecto, pero sirve como filtro mínimo.
    """
    a_terms = set(re.findall(r"\b\w+\b", (answer or "").lower()))
    c_terms = set(re.findall(r"\b\w+\b", (context or "").lower()))
    if not a_terms:
        return 0.0
    return len(a_terms & c_terms) / max(1, len(a_terms))


class LocalRAG:
    """
    Modo local:
    - Evidence: BM25 (BaselineStore)
    - Redacción: Ollama (LLM local)
    - Fallback: baseline extractivo

    Filosofía:
    - El LLM NO decide evidencia.
    - Si no hay evidencia suficiente (gate), NO se llama al LLM.
    - Para preguntas definicionales ("¿qué es...?"), exige señales de definición.
    - Validación post-LLM para bloquear meta/respuestas “humo”.
    """

    def __init__(
        self,
        pdf_path: str,
        k: int = 5,
        min_best_score: float = 0.15,
        max_context_chars: int = 9000,
    ):
        self.k = k
        self.min_best_score = min_best_score
        self.max_context_chars = max_context_chars

        # Baseline para retrieval + fallback
        self.baseline = BaselineRAG(pdf_path=pdf_path, debug=False)

        # Generador local (Ollama)
        self.generator = OllamaClient()

    def _build_context(self, chunks) -> str:
        """
        Construye el contexto con delimitadores fuertes.
        Además lo limita a max_context_chars para evitar prompts gigantes.
        """
        parts: List[str] = []
        for i, c in enumerate(chunks, 1):
            txt = _normalize(getattr(c, "text", str(c)))
            if not txt:
                continue
            parts.append(f"[DOC {i}]\n{txt}")

        context = "\n\n".join(parts)
        if len(context) <= self.max_context_chars:
            return context

        # Recorte seguro
        context = context[: self.max_context_chars].rsplit(" ", 1)[0]
        return context

    def _build_prompt(self, context: str, question: str) -> str:
        # Prompt cerrado, sin conversación.
        return f"""INSTRUCCIONES (obligatorias):
1) Responde SOLO la pregunta.
2) Usa SOLO el CONTEXTO. No inventes. No uses conocimiento externo.
3) No escribas explicaciones meta (ej: "como modelo", "no tengo acceso", etc.).
4) Si el CONTEXTO no contiene la respuesta, responde EXACTAMENTE:
{NO_EVIDENCE}

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA (máx 5 líneas):
""".strip()

    def ask(self, question: str) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            return {"respuesta": NO_EVIDENCE, "fuentes": []}

        # 1) Retrieval BM25 con scores
        hits: List[Tuple[Any, float]] = self.baseline.store.search(question, k=self.k)

        if not hits:
            return {"respuesta": NO_EVIDENCE, "fuentes": []}

        best_score = hits[0][1] if hits else 0.0

        # 2) Gate determinístico: si BM25 está débil, NO llamamos al LLM
        if best_score < self.min_best_score:
            return {"respuesta": NO_EVIDENCE, "fuentes": []}

        chunks = [c for c, _ in hits]

        # 2.5) Gate definicional: evita confundir "mención" con "definición"
        q = question.lower()
        is_definition = (
            "¿qué es" in q or "que es" in q or
            "definición" in q or "definicion" in q or
            "define" in q or "defina" in q
        )

        if is_definition:
            joined = " ".join(getattr(c, "text", str(c)).lower() for c in chunks)
            definers = [
                "se define", "se entiende", "consiste", "se considera", "se refiere",
                "definición", "definicion"
            ]
            if not any(d in joined for d in definers):
                return {"respuesta": NO_EVIDENCE, "fuentes": []}

        context = self._build_context(chunks)
        prompt = self._build_prompt(context, question)

        # 3) Generación local + validación
        try:
            answer = self.generator.generate(prompt)
            answer = (answer or "").strip()

            if not answer:
                raise ValueError("LLM devolvió respuesta vacía")

            # Si el modelo no respeta el contrato o mete meta/disclaimer
            if _looks_like_meta_or_disclaimer(answer):
                raise ValueError("Respuesta meta/disclaimer detectada")

            # Si el modelo respondió NO_EVIDENCE, ok (y forzamos fuentes vacías)
            if answer.strip() == NO_EVIDENCE:
                return {"respuesta": NO_EVIDENCE, "fuentes": []}

            # Heurística rápida de grounding: overlap mínimo con el contexto
            if _overlap_ratio(answer, context) < 0.05:
                raise ValueError("Bajo solapamiento respuesta-contexto")

            return {
                "respuesta": answer,
                "fuentes": [getattr(c, "text", str(c)) for c in chunks],
            }

        except Exception:
            # 4) Fallback fuerte a baseline extractivo
            return self.baseline.ask(question)
