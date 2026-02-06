# vector_store.py
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from backend.Pdf_loader import cargar_pdf


# Cargar variables de entorno
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paso 1: Cargar fragmentos del PDF
documentos = cargar_pdf("Doc chatbot.pdf")
print(f"✅ Fragmentos cargados: {len(documentos)}")

# Paso 2: Crear los embeddings con Gemini (Google Generative AI)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# Paso 3: Almacenar en Chroma
chroma_db = Chroma.from_documents(
    documents=documentos,
    embedding=embeddings,
    persist_directory="chroma_db_riesgos"
)

# Paso 4: Guardar en disco
chroma_db.persist()
print("✅ Vector store guardado en 'chroma_db_riesgos'")
