import os
import logging
from pathlib import Path
from typing import Any, Dict

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
    - llm: Gemini + Chroma (opcional, requiere API key)

    Control de modo:
    - Variable de entorno RAG_MODE = "baseline" | "llm"
      Si no está definida, selecciona automáticamente.
    """

    def __init__(self, persist_directory: str = "chroma_db_riesgos", temperature: float = 0.3, model: str = "gemini-1.5-flash"):
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.model = model

        self.vectorstore = None
        self.qa_chain = None
        self.baseline = None
        self.embedding_function = None
        self.llm = None

        # Cargar variables de entorno
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

        # ✅ Ruta robusta al PDF
        self.pdf_path = _default_pdf_path()

        # 🔑 Selección de modo (forzado o automático)
        forced_mode = os.getenv("RAG_MODE", "").strip().lower()  # "baseline" o "llm"
        if forced_mode in {"baseline", "llm"}:
            self.mode = forced_mode
        else:
            # Auto: si hay API key y existe el vectorstore -> llm, si no -> baseline
            if self.google_api_key and Path(self.persist_directory).exists():
                self.mode = "llm"
            else:
                self.mode = "baseline"

        logger.info(f"🧩 Modo seleccionado: {self.mode}")

        # 🟢 BASELINE (sin tokens)
        if self.mode == "baseline":
            self.baseline = BaselineRAG(pdf_path=self.pdf_path, debug=False)
            logger.info("✅ Baseline inicializado (BM25 + extractivo).")
            return

        # 🔵 LLM (opcional)
        self._setup_components()

    def _setup_components(self):
        """Configura los componentes del modo LLM (Gemini + Chroma)."""
        try:
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
            # 🔥 En vez de romper el sistema, degradamos a baseline
            logger.error(f"❌ Error al inicializar componentes LLM: {e}")
            logger.info("↩️ Fallback automático a baseline (sin tokens).")

            self.mode = "baseline"
            self.baseline = BaselineRAG(pdf_path=self.pdf_path, debug=False)

            self.vectorstore = None
            self.qa_chain = None
            self.embedding_function = None
            self.llm = None

            logger.info("✅ Baseline inicializado tras fallo en LLM.")

    def consultar(self, pregunta: str, mostrar_fuentes: bool = False) -> Dict[str, Any]:
        """
        Realiza una consulta al sistema.

        Returns:
            dict: {"respuesta": str, "fuentes": list}
        """
        try:
            logger.info(f"🔍 Procesando consulta: {pregunta[:50]}...")

            # ✅ Baseline
            if getattr(self, "mode", "llm") == "baseline":
                return self.baseline.ask(pregunta)

            # ✅ LLM
            if self.qa_chain is None:
                raise RuntimeError("qa_chain no está inicializada en modo llm.")

            respuesta = self.qa_chain.invoke({"query": pregunta})

            resultado = {
                "respuesta": respuesta["result"],
                "fuentes": respuesta.get("source_documents", [])
            }

            if mostrar_fuentes and resultado["fuentes"]:
                logger.info(f"📄 Encontradas {len(resultado['fuentes'])} fuentes relevantes")

            return resultado

        except Exception as e:
            logger.error(f"❌ Error al procesar consulta (modo {getattr(self, 'mode', '?')}): {e}")

            # 🔁 Si falló LLM (429/cuota/etc.), degradar a baseline automáticamente
            if getattr(self, "mode", "llm") == "llm":
                logger.info("↩️ Fallback a baseline por error en LLM.")
                try:
                    self.mode = "baseline"
                    if self.baseline is None:
                        self.baseline = BaselineRAG(pdf_path=self.pdf_path, debug=False)
                    return self.baseline.ask(pregunta)
                except Exception as e2:
                    logger.error(f"❌ También falló baseline: {e2}")

            return {"respuesta": f"❌ Error al procesar la consulta: {str(e)}", "fuentes": []}

    def buscar_documentos_similares(self, consulta: str, k: int = 3):
        """
        Busca documentos similares sin generar respuesta.
        Funciona en baseline y en llm.
        """
        try:
            # ✅ Baseline: devolvemos los chunks más relevantes del BM25
            if getattr(self, "mode", "llm") == "baseline":
                hits = self.baseline.store.search(consulta, k=k)  # [(Chunk, score), ...]
                return [chunk for (chunk, _score) in hits]

            # ✅ LLM: búsqueda semántica
            if self.vectorstore is None:
                raise RuntimeError("vectorstore no está inicializado en modo llm.")
            return self.vectorstore.similarity_search(consulta, k=k)

        except Exception as e:
            logger.error(f"❌ Error en búsqueda de similitud: {e}")
            return []

    def mostrar_fuentes(self, fuentes):
        """Muestra las fuentes de información de manera formateada."""
        if not fuentes:
            print("📄 No se encontraron fuentes específicas.")
            return

        print(f"\n📚 Fuentes consultadas ({len(fuentes)}):")
        print("-" * 50)

        for i, doc in enumerate(fuentes, 1):
            content = getattr(doc, "page_content", None)
            if content is None:
                # baseline chunk
                content = getattr(doc, "text", str(doc))

            contenido = content[:200] + "..." if len(content) > 200 else content
            print(f"{i}. {contenido}")

            meta = getattr(doc, "metadata", None)
            if meta:
                print(f"   📋 Metadata: {meta}")
            print()
