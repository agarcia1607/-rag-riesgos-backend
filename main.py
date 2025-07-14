# main.py - API FastAPI para RAG con Gemini
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from query_wrapper import ChatbotRiesgos
from fastapi.middleware.cors import CORSMiddleware








# Crear app
app = FastAPI()
chatbot = ChatbotRiesgos()




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # o ['https://rag-riesgos.vercel.app']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CORS (para frontend React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de entrada
class Pregunta(BaseModel):
    texto: str

@app.post("/preguntar")
async def preguntar(pregunta: Pregunta):
    respuesta = chatbot.consultar(pregunta.texto, mostrar_fuentes=True)
    return {
        "respuesta": respuesta["respuesta"],
        "fuentes": [doc.page_content for doc in respuesta["fuentes"]]
    }

@app.get("/")
def root():
    return {"mensaje": "API de RAG de Riesgos activa 🚀"}
