import re
import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ftfy import fix_text


def _clean(text: str) -> str:
    text = fix_text(text)               # ✅ arregla mojibake (SegÃºn -> Según, â¢ -> •)
    text = text.replace("\xa0", " ")    # NBSP
    text = re.sub(r"\s+", " ", text)    # espacios múltiples
    return text.strip()


def cargar_pdf(ruta_pdf: str) -> list[Document]:
    doc = fitz.open(ruta_pdf)
    pages_text = []

    for i, pagina in enumerate(doc):
        raw = pagina.get_text()
        cleaned = _clean(raw)
        if cleaned:
            # separador fuerte entre páginas
            pages_text.append(f"[PAGE {i+1}]\n{cleaned}")

    texto_completo = "\n\n".join(pages_text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(texto_completo)

    # ✅ guarda metadata (página aproximada) para futuro; por ahora igual sirve
    documentos = [Document(page_content=c, metadata={"source": ruta_pdf}) for c in chunks]
    return documentos


if __name__ == "__main__":
    ruta_pdf = "Doc chatbot.pdf"
    documentos = cargar_pdf(ruta_pdf)

    print(f"✅ Total de fragmentos: {len(documentos)}\n")
    for i, d in enumerate(documentos[:5]):
        print(f"[{i}] {d.page_content}\n")
