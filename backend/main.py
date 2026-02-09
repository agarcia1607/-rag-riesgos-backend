from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn



from backend.query_wrapper import ChatbotRiesgos
from backend.baseline_rag import BaselineRAG




from fastapi.responses import JSONResponse

app = FastAPI(default_response_class=JSONResponse)






@app.get("/health")
def health():
    return {"status": "ok"}


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod limita el dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Pregunta(BaseModel):
    texto: str

# Inicializa una sola vez (no por request)
chatbot = ChatbotRiesgos()

def _normalize_fuentes(raw_fuentes):
    """
    En LLM: raw_fuentes suelen ser Document con .page_content
    En baseline: raw_fuentes suelen ser strings (ya texto)
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

@app.post("/preguntar")
async def preguntar(pregunta: Pregunta):
    r = chatbot.consultar(pregunta.texto, mostrar_fuentes=True)
    return {
        "respuesta": r.get("respuesta", ""),
        "fuentes": _normalize_fuentes(r.get("fuentes", [])),
        "mode": getattr(chatbot, "mode", "unknown"),
    }

@app.get("/")
def root():
    return {"mensaje": "API de RAG de Riesgos activa 🚀", "mode": getattr(chatbot, "mode", "unknown")}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
