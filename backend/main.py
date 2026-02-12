from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn

from backend.query_wrapper import ChatbotRiesgos

app = FastAPI(default_response_class=JSONResponse)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod limita el dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."


class Pregunta(BaseModel):
    texto: str
    mode: str | None = None   # ✅ baseline/local/llm (opcional)


# Inicializa una sola vez (no por request)
chatbot = ChatbotRiesgos()


def _normalize_fuentes(raw_fuentes):
    """
    En LLM: raw_fuentes pueden ser Document con .page_content
    En baseline/local: raw_fuentes suelen ser strings
    """
    fuentes = []
    for f in raw_fuentes or []:
        if isinstance(f, str):
            fuentes.append(f)
        elif hasattr(f, "page_content"):
            fuentes.append(f.page_content)
        elif hasattr(f, "text"):
            fuentes.append(f.text)
        else:
            fuentes.append(str(f))
    return fuentes


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preguntar")
async def preguntar(pregunta: Pregunta):
    r = chatbot.consultar(pregunta.texto, mostrar_fuentes=True, mode=pregunta.mode)

    # normaliza fuentes pero no borres retrieved/flags
    r["fuentes"] = _normalize_fuentes(r.get("fuentes", []))
    r.setdefault("retrieved", [])
    r.setdefault("no_evidence", r.get("respuesta", "") == NO_EVIDENCE)
    r.setdefault("used_fallback", False)
    r.setdefault("gate_reason", None)
    r.setdefault("mode", pregunta.mode or getattr(chatbot, "mode", "unknown"))
    return r



@app.get("/")
def root():
    return {"mensaje": "API de RAG de Riesgos activa 🚀", "mode": getattr(chatbot, "mode", "unknown")}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
