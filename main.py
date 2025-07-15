# main.py - API FastAPI para RAG con Gemini

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from query_wrapper import ChatbotRiesgos
import os
import uvicorn

# Crear app
app = FastAPI()
chatbot = ChatbotRiesgos()

# Configurar CORS para permitir conexión desde Vercel (puedes especificar el dominio si lo prefieres)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto por ["https://rag-riesgos-frontend.vercel.app"] para más seguridad
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class Pregunta(BaseModel):
    texto: str

# Endpoint principal
@app.post("/preguntar")
async def preguntar(pregunta: Pregunta):
    respuesta = chatbot.consultar(pregunta.texto, mostrar_fuentes=True)
    return {
        "respuesta": respuesta["respuesta"],
        "fuentes": [doc.page_content for doc in respuesta["fuentes"]]
    }

# Endpoint de prueba
@app.get("/")
def root():
    return {"mensaje": "API de RAG de Riesgos activa 🚀"}

# Ejecutar localmente o en Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
