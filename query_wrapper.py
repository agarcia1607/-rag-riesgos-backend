import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatbotRiesgos:
    """
    Chatbot para consultas sobre riesgos usando Gemini y Chroma vector store.
    
    Modelos disponibles:
    - gemini-1.5-flash: Rápido y eficiente (recomendado) - 1,500 req/día gratis
    - gemini-1.5-pro: Mejor calidad de respuesta - 50 req/día gratis
    - gemini-2.0-flash: Modelo más reciente con capacidades multimodales
    - gemini-2.5-flash: Mejor rendimiento precio-calidad
    - gemini-2.5-pro: Modelo más avanzado para razonamiento complejo
    """
    
    def __init__(self, persist_directory="chroma_db_riesgos", temperature=0.3, model="gemini-1.5-flash"):
        """
        Inicializa el chatbot.
        
        Args:
            persist_directory: Directorio donde está guardado el vector store
            temperature: Temperatura para el modelo (0.0 - 1.0)
            model: Modelo de Gemini a usar (por defecto: gemini-1.5-flash)
        """
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.model = model  # gemini-1.5-flash por defecto (1,500 req/día gratis)
        self.vectorstore = None
        self.qa_chain = None
        
        # Cargar variables de entorno
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Validar configuración
        self._validate_config()
        
        # Inicializar componentes
        self._setup_components()
    
    def _validate_config(self):
        """Valida la configuración inicial."""
        if not self.google_api_key:
            raise ValueError("❌ GOOGLE_API_KEY no encontrada en las variables de entorno")
        
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"❌ Vector store no encontrado en: {self.persist_directory}")
    
    def _setup_components(self):
        """Configura los componentes del chatbot."""
        try:
            # Inicializar función de embeddings
            logger.info("🔧 Inicializando embeddings...")
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.google_api_key
            )
            
            # Cargar vectorstore
            logger.info("📚 Cargando vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function
            )
            
            # Inicializar modelo Gemini
            logger.info(f"🤖 Inicializando modelo Gemini: {self.model}")
            
            # Validar modelo disponible
            modelos_disponibles = [
                "gemini-1.5-flash",
                "gemini-1.5-pro", 
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-2.5-pro"
            ]
            
            if self.model not in modelos_disponibles:
                logger.warning(f"⚠️  Modelo {self.model} no reconocido. Usando gemini-1.5-flash por defecto.")
                self.model = "gemini-1.5-flash"
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.google_api_key,
                temperature=self.temperature
            )
            
            # Crear prompt personalizado
            self.prompt_template = PromptTemplate(
                template="""
                Eres un asistente experto en análisis de riesgos. Responde de manera clara y precisa basándote en la información proporcionada.
                
                Contexto: {context}
                
                Pregunta: {question}
                
                Respuesta detallada:
                """,
                input_variables=["context", "question"]
            )
            
            # Crear cadena QA con recuperación
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
            
            logger.info("✅ Chatbot inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error al inicializar componentes: {e}")
            raise
    
    def consultar(self, pregunta: str, mostrar_fuentes: bool = False):
        """
        Realiza una consulta al chatbot.
        
        Args:
            pregunta: Pregunta a realizar
            mostrar_fuentes: Si mostrar documentos fuente
        
        Returns:
            dict: Respuesta con resultado y fuentes
        """
        try:
            logger.info(f"🔍 Procesando consulta: {pregunta[:50]}...")
            
            # Realizar consulta
            respuesta = self.qa_chain.invoke({"query": pregunta})
            
            resultado = {
                "respuesta": respuesta["result"],
                "fuentes": respuesta.get("source_documents", [])
            }
            
            if mostrar_fuentes and resultado["fuentes"]:
                logger.info(f"📄 Encontradas {len(resultado['fuentes'])} fuentes relevantes")
            
            return resultado
            
        except Exception as e:
            logger.error(f"❌ Error al procesar consulta: {e}")
            return {
                "respuesta": f"❌ Error al procesar la consulta: {str(e)}",
                "fuentes": []
            }
    
    def buscar_documentos_similares(self, consulta: str, k: int = 3):
        """
        Busca documentos similares sin generar respuesta.
        
        Args:
            consulta: Texto a buscar
            k: Número de documentos a retornar
        
        Returns:
            list: Lista de documentos similares
        """
        try:
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
            # Mostrar fragmento del contenido
            contenido = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"{i}. {contenido}")
            
            # Mostrar metadata si está disponible
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"   📋 Metadata: {doc.metadata}")
            print()

def main():
    """Función principal con loop interactivo."""
    print("🤖 Chatbot de Riesgos - Gemini + Chroma")
    print("=" * 50)
    print("Comandos disponibles:")
    print("  - 'salir', 'exit', 'quit': Terminar")
    print("  - 'fuentes': Mostrar fuentes de la última consulta")
    print("  - 'buscar [texto]': Buscar documentos similares")
    print("=" * 50)
    
    try:
        # Inicializar chatbot
        chatbot = ChatbotRiesgos()
        ultima_respuesta = None
        
        while True:
            try:
                pregunta = input("\n❓ Pregunta: ").strip()
                
                # Comandos de salida
                if pregunta.lower() in {"salir", "exit", "quit", ""}:
                    print("👋 ¡Hasta luego!")
                    break
                
                # Mostrar fuentes de última consulta
                if pregunta.lower() == "fuentes":
                    if ultima_respuesta and ultima_respuesta.get("fuentes"):
                        chatbot.mostrar_fuentes(ultima_respuesta["fuentes"])
                    else:
                        print("❌ No hay fuentes disponibles de la última consulta.")
                    continue
                
                # Búsqueda de documentos similares
                if pregunta.lower().startswith("buscar "):
                    texto_busqueda = pregunta[7:].strip()
                    if texto_busqueda:
                        documentos = chatbot.buscar_documentos_similares(texto_busqueda)
                        if documentos:
                            print(f"\n🔍 Documentos similares encontrados ({len(documentos)}):")
                            chatbot.mostrar_fuentes(documentos)
                        else:
                            print("❌ No se encontraron documentos similares.")
                    continue
                
                # Realizar consulta normal
                ultima_respuesta = chatbot.consultar(pregunta)
                
                # Mostrar respuesta
                print(f"\n🧠 Respuesta:\n{ultima_respuesta['respuesta']}")
                
                # Indicar si hay fuentes disponibles
                if ultima_respuesta.get("fuentes"):
                    print(f"\n💡 Tip: Escribe 'fuentes' para ver las {len(ultima_respuesta['fuentes'])} fuentes consultadas")
                
            except KeyboardInterrupt:
                print("\n\n👋 Saliendo...")
                break
            except Exception as e:
                logger.error(f"❌ Error en el loop principal: {e}")
                print(f"❌ Error inesperado: {e}")
    
    except Exception as e:
        logger.error(f"❌ Error al inicializar el chatbot: {e}")
        print(f"❌ No se pudo inicializar el chatbot: {e}")

if __name__ == "__main__":
    main()