from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# ✅ Imports internos como paquete "backend"
from backend.baseline_rag import BaselineRAG

# (LLM opcional)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Contrato estándar (baseline/local/llm)
NO_EVIDENCE_STD = "No se encontró evidencia suficiente en los documentos."


def _default_pdf_path() -> str:
    """
    Devuelve la ruta absoluta al PDF dentro de /data, independientemente
    de desde dónde se ejecute el programa.
    Estructura esperada:
      repo_root/
        data/Doc chatbot.pdf
        backend/query_wrapper.py
    """
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / "data" / "Doc chatbot.pdf")


class ChatbotRiesgos:
    """
    Chatbot RAG para consultas sobre riesgos.

    Modos:
    - baseline: BM25 + extractivo (sin tokens, reproducible)
    - local: BM25 + Ollama (solo redacción/síntesis; evidencia la decide BM25)
    - llm: Gemini + Chroma (opcional, requiere API key)

    Control de modo:
    - Variable de entorno RAG_MODE = "baseline" | "local" | "llm"
      Si no está definida, selecciona automáticamente.

    Además:
    - consultar(..., mode="baseline"/"local"/"llm") permite forzar por request,
      SIN cambiar el modo global del objeto.
    """

    def __init__(
        self,
        persist_directory: str = "chroma_db_riesgos",
        temperature: float = 0.3,
        model: str = "gemini-1.5-flash",
    ):
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.model = model

        self.vectorstore = None
        self.qa_chain = None
        self.baseline = None
        self.local = None
        self.embedding_function = None
        self.llm = None

        # Cargar variables de entorno
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        # ✅ Ruta robusta al PDF
        self.pdf_path = _default_pdf_path()

        # 🔑 Selección de modo (forzado o automático)
        forced_mode = os.getenv("RAG_MODE", "").strip().lower()  # "baseline" | "local" | "llm"
        if forced_mode in {"baseline", "local", "llm"}:
            self.mode = forced_mode
        else:
            if self.google_api_key and Path(self.persist_directory).exists():
                self.mode = "llm"
            elif os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL"):
                self.mode = "local"
            else:
                self.mode = "baseline"

        logger.info(f"🧩 Modo seleccionado (default): {self.mode}")

        # ✅ Baseline SIEMPRE disponible (y base para local)
        self.baseline = BaselineRAG(pdf_path=self.pdf_path, debug=False)

        # Lazy: no levantamos todo si no toca
        if self.mode == "local":
            self._ensure_local()
            if self.local is None:
                self.mode = "baseline"
                logger.info("✅ Baseline activo tras fallo en local.")

        if self.mode == "llm":
            self._ensure_llm()
            if self.qa_chain is None:
                self.mode = "baseline"
                logger.info("✅ Baseline activo tras fallo en llm.")

        logger.info("✅ Sistema listo.")

    def _ensure_local(self) -> None:
        """Inicializa LocalRAG (lazy) si no está."""
        if self.local is not None:
            return
        try:
            from backend.local_rag import LocalRAG
            self.local = LocalRAG(pdf_path=self.pdf_path)
            logger.info("🟣 Modo local inicializado (lazy, BM25 + Ollama).")
        except Exception as e:
            logger.error(f"❌ Error al inicializar modo local: {e}")
            self.local = None

    def _ensure_llm(self) -> None:
        """Inicializa LLM (Gemini + Chroma) (lazy) si no está."""
        if self.qa_chain is not None and self.vectorstore is not None:
            return
        self._setup_components()

    def _setup_components(self) -> None:
        """Configura los componentes del modo LLM (Gemini + Chroma)."""
        try:
            if not self.google_api_key:
                raise RuntimeError("GOOGLE_API_KEY no está configurada.")

            logger.info("🔧 Inicializando embeddings...")
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )

            logger.info("📚 Cargando vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )

            logger.info(f"🤖 Inicializando modelo Gemini: {self.model}")

            modelos_disponibles = [
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ]
            if self.model not in modelos_disponibles:
                logger.warning(f"⚠️ Modelo {self.model} no reconocido. Usando gemini-1.5-flash por defecto.")
                self.model = "gemini-1.5-flash"

            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.google_api_key,
                temperature=self.temperature
            )

            self.prompt_template = PromptTemplate(
                template=(
                    "Eres un asistente experto en análisis de riesgos. "
                    "Responde de manera clara y precisa basándote en la información proporcionada.\n\n"
                    "Contexto: {context}\n\n"
                    "Pregunta: {question}\n\n"
                    "Respuesta detallada:"
                ),
                input_variables=["context", "question"]
            )

            logger.info("🔗 Configurando cadena QA...")
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )

            logger.info("✅ Modo LLM inicializado correctamente.")

        except Exception as e:
            logger.error(f"❌ Error al inicializar componentes LLM: {e}")
            self.vectorstore = None
            self.qa_chain = None
            self.embedding_function = None
            self.llm = None

    def consultar(self, pregunta: str, mostrar_fuentes: bool = False, mode: str | None = None) -> Dict[str, Any]:
        """
        Realiza una consulta al sistema.

        mode (opcional): "baseline" | "local" | "llm"
        Si viene, sobreescribe el modo actual SOLO para esta consulta.
        """
        pregunta = (pregunta or "").strip()
        if not pregunta:
            return {
                "mode": mode or getattr(self, "mode", "baseline"),
                "respuesta": NO_EVIDENCE_STD,
                "fuentes": [],
                "retrieved": [],
                "no_evidence": True,
                "used_fallback": False,
                "gate_reason": "empty_question",
            }

        effective_mode = (mode or getattr(self, "mode", "baseline") or "baseline").strip().lower()
        logger.info(f"🔍 Consulta modo={effective_mode}: {pregunta[:80]}")

        # ✅ Baseline
        if effective_mode == "baseline":
            out = self.baseline.ask(pregunta)
            out["mode"] = "baseline"
            return out

        # ✅ Local (lazy init)
        if effective_mode == "local":
            self._ensure_local()
            if self.local is None:
                # degradación segura al baseline (pero lo reportamos)
                out = self.baseline.ask(pregunta)
                out["mode"] = "local"
                out["used_fallback"] = True
                out["gate_reason"] = "fallback(local_init_failed)"
                return out

            out = self.local.ask(pregunta)
            out["mode"] = "local"
            return out

        # ✅ LLM remoto (lazy init)
        if effective_mode == "llm":
            self._ensure_llm()
            if self.qa_chain is None:
                out = self.baseline.ask(pregunta)
                out["mode"] = "llm"
                out["used_fallback"] = True
                out["gate_reason"] = "fallback(llm_init_failed)"
                return out

            respuesta = self.qa_chain.invoke({"query": pregunta})
            resultado = {
                "mode": "llm",
                "respuesta": respuesta.get("result", "") or "",
                "fuentes": respuesta.get("source_documents", []),
                # (si luego quieres: aquí metemos retrieved semántico)
                "retrieved": [],
                "no_evidence": False,
                "used_fallback": False,
                "gate_reason": None,
            }

            if not resultado["respuesta"]:
                resultado["respuesta"] = NO_EVIDENCE_STD
                resultado["no_evidence"] = True
                resultado["gate_reason"] = "llm_empty_answer"

            if mostrar_fuentes and resultado["fuentes"]:
                logger.info(f"📄 Encontradas {len(resultado['fuentes'])} fuentes (llm).")

            return resultado

        # modo inválido
        out = self.baseline.ask(pregunta)
        out["mode"] = effective_mode
        out["used_fallback"] = True
        out["gate_reason"] = f"fallback(invalid_mode:{effective_mode})"
        return out

    def buscar_documentos_similares(self, consulta: str, k: int = 3):
        """
        Busca documentos similares sin generar respuesta.
        Funciona en baseline, local y en llm.
        """
        try:
            # Baseline/Local: BM25
            if getattr(self, "mode", "baseline") in {"baseline", "local"}:
                hits = self.baseline.store.search(consulta, k=k)  # [(Chunk, score), ...]
                return [chunk for (chunk, _score) in hits]

            # LLM: semántico
            if self.vectorstore is None:
                raise RuntimeError("vectorstore no está inicializado en modo llm.")
            return self.vectorstore.similarity_search(consulta, k=k)

        except Exception as e:
            logger.error(f"❌ Error en búsqueda de similitud: {e}")
            return []

    def mostrar_fuentes(self, fuentes) -> None:
        """Muestra las fuentes de información de manera formateada."""
        if not fuentes:
            print("📄 No se encontraron fuentes específicas.")
            return

        print(f"\n📚 Fuentes consultadas ({len(fuentes)}):")
        print("-" * 50)

        for i, doc in enumerate(fuentes, 1):
            content = getattr(doc, "page_content", None)
            if content is None:
                content = getattr(doc, "text", str(doc))

            contenido = content[:200] + "..." if len(content) > 200 else content
            print(f"{i}. {contenido}")

            meta = getattr(doc, "metadata", None)
            if meta:
                print(f"   📋 Metadata: {meta}")
            print()
