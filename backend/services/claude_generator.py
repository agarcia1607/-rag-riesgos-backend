# backend/services/claude_generator.py
import os
import anthropic
from typing import Optional

SYSTEM_PROMPT = """Eres un asistente de análisis de riesgos.
REGLA ABSOLUTA: Solo puedes responder usando información EXPLÍCITAMENTE presente en los fragmentos dados.
Si la pregunta es vaga, general, o la respuesta exacta no está en los fragmentos, responde ÚNICAMENTE con: NO_EVIDENCE
No inferras, no extrapoles, no uses conocimiento externo bajo ninguna circunstancia."""

def generate_answer(query: str, context_chunks: list[dict]) -> Optional[str]:
    if not context_chunks:
        return None

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    context_text = "\n\n---\n\n".join(
        f"[Fragmento {i+1}]\n{chunk.get('text', chunk.get('content', ''))}"
        for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""Contexto de los documentos:
{context_text}
---
Pregunta: {query}

Instrucciones:
- Si la pregunta es vaga o general y no tiene una respuesta específica en el contexto, responde: NO_EVIDENCE
- Si la respuesta exacta no está en los fragmentos, responde: NO_EVIDENCE
- Solo responde si hay evidencia textual directa y específica."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )
        answer = response.content[0].text.strip()

        # Gate principal: señal explícita
        if answer.strip() == "NO_EVIDENCE":
            return None

        # Gate fallback: frases de abstención
        abstention_phrases = [
            "no puedo proporcionar", "no puedo determinar", "no puedo identificar",
            "no puedo responder", "no tengo información",
            "no se menciona", "no está disponible en el contexto",
            "no encuentro evidencia", "no hay información suficiente",
            "no_evidence"
        ]
        if any(phrase in answer.lower() for phrase in abstention_phrases):
            return None

        return answer

    except Exception as e:
        print(f"[Claude] Error: {e}")
        return None