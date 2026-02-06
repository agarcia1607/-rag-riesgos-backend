# Pdf_loader.py
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def cargar_pdf(ruta_pdf: str) -> list[Document]:
    doc = fitz.open(ruta_pdf)
    texto_completo = ""

    for pagina in doc:
        texto_completo += pagina.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(texto_completo)

    documentos = [Document(page_content=c) for c in chunks]
    return documentos

if __name__ == "__main__":
    ruta_pdf = "Doc chatbot.pdf"
    documentos = cargar_pdf(ruta_pdf)
    
    print(f"✅ Total de fragmentos: {len(documentos)}\n")
    for i, doc in enumerate(documentos[:5]):
        print(f"[{i}] {doc.page_content}\n")
