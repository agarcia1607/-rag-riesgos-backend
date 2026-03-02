from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn
from dotenv import load_dotenv

from backend.query_wrapper import ChatbotRiesgos

# =====================================================
# CARGA DE VARIABLES DE ENTORNO (.env global)
# =====================================================
load_dotenv()

app = FastAPI(default_response_class=JSONResponse)

# =====================================================
# CORS CONFIGURABLE DESDE .env
# =====================================================
def parse_allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    parts = [p.strip() for p in raw.replace(" ", ",").split(",")]
    return [p for p in parts if p]


ALLOWED_ORIGINS = parse_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,  # si no usas cookies, debe ser False
    allow_methods=["*"],
    allow_headers=["*"],
)

NO_EVIDENCE = "No se encontró evidencia suficiente en los documentos."


class Pregunta(BaseModel):
    texto: str
    mode: str | None = None  # baseline | local | llm
    top_k: int | None = None  # opcional: solo si el backend lo soporta


# =====================================================
# INICIALIZACIÓN GLOBAL (NO POR REQUEST)
# =====================================================
chatbot = ChatbotRiesgos()


def _normalize_fuentes(raw_fuentes):
    """
    En LLM: raw_fuentes pueden ser Document con .page_content
    En baseline/local: raw_fuentes suelen ser strings
    """
    fuentes: list[str] = []
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


# =====================================================
# HELPERS: PROD GUARDRAILS
# =====================================================
def is_render() -> bool:
    # Render suele setear RENDER=true y/o RENDER_SERVICE_ID
    return os.getenv("RENDER", "").lower() == "true" or os.getenv("RENDER_SERVICE_ID") is not None


def should_force_baseline(requested_mode: str | None) -> bool:
    """
    En Render NO hay Ollama local.
    Para que producción no se caiga: en Render forzamos baseline si piden local/llm.
    """
    rm = (requested_mode or "local").lower()
    return bool(is_render() and rm in ("local", "llm"))


def _safe_contract(
    *,
    respuesta: str,
    fuentes=None,
    retrieved=None,
    mode: str,
    requested_mode: str,
    no_evidence: bool | None = None,
    used_fallback: bool = False,
    gate_reason: str | None = None,
    error: str | None = None,
):
    data = {
        "respuesta": respuesta,
        "fuentes": _normalize_fuentes(fuentes or []),
        "retrieved": retrieved or [],
        "mode": mode,
        "requested_mode": requested_mode,
        "no_evidence": bool(no_evidence) if no_evidence is not None else (respuesta == NO_EVIDENCE),
        "used_fallback": bool(used_fallback),
        "gate_reason": gate_reason,
    }
    if error is not None:
        data["error"] = error
    return data


# =====================================================
# ENDPOINTS
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preguntar")
async def preguntar(pregunta: Pregunta):
    requested_mode = (pregunta.mode or getattr(chatbot, "mode", "unknown")).lower()

    effective_mode = requested_mode
    if should_force_baseline(requested_mode):
        effective_mode = "baseline"

    try:
        # ✅ Llamada segura: NO pasamos top_k a menos que sepamos que consultar() lo soporta.
        # Como no lo sabemos, lo omitimos para evitar 500s en producción.
        r = chatbot.consultar(
            pregunta.texto,
            mostrar_fuentes=True,
            mode=effective_mode
        )

        # Normalización + contrato estable
        r["fuentes"] = _normalize_fuentes(r.get("fuentes", []))
        r.setdefault("retrieved", [])
        r.setdefault("used_fallback", False)
        r.setdefault("gate_reason", None)
        r.setdefault("no_evidence", (r.get("respuesta", "") == NO_EVIDENCE))

        # Debug: reporta ambos
        r["mode"] = effective_mode
        r["requested_mode"] = requested_mode

        # Asegura llaves base aunque el wrapper devuelva algo raro
        r.setdefault("respuesta", "")
        r.setdefault("fuentes", [])
        r.setdefault("retrieved", [])

        return r

    except Exception as e:
        # ✅ Nunca devuelvas plain-text 500; responde JSON con contrato estable
        return _safe_contract(
            respuesta=f"❌ Error al procesar la consulta: {str(e)}",
            fuentes=[],
            retrieved=[],
            mode=effective_mode,
            requested_mode=requested_mode,
            no_evidence=True,
            used_fallback=True,
            gate_reason="exception",
            error=type(e).__name__,
        )


@app.get("/")
def root():
    return {
        "mensaje": "API de RAG de Riesgos activa 🚀",
        "mode": getattr(chatbot, "mode", "unknown"),
        "allowed_origins": ALLOWED_ORIGINS,
        "is_render": is_render(),
    }


# =====================================================
# ENTRYPOINT
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)