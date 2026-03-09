import re
import fitz
from ftfy import fix_text
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _clean(text: str) -> str:
    text = fix_text(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _looks_like_toc(text: str) -> bool:
    """
    Detecta chunks tipo tabla de contenido / índice.
    """
    t = (text or "").strip().lower()

    if not t:
        return True

    # muchas líneas de puntos / estructura de índice
    if "................................................................" in t:
        return True

    # encabezados típicos de índice
    toc_terms = [
        "datos generales",
        "territorialidad",
        "medios de transporte",
        "riesgos cubiertos",
        "deducibles",
        "medidas de seguridad",
        "procedimiento en caso de siniestro",
    ]

    hits = sum(1 for term in toc_terms if term in t)
    if hits >= 3:
        return True

    return False


def _is_noise(text: str) -> bool:
    t = (text or "").strip()

    if not t:
        return True

    # chunks demasiado cortos
    if len(t) < 40:
        return True

    # marcador de página aislado
    if re.fullmatch(r"\[page\s+\d+\]", t, re.IGNORECASE):
        return True

    # solo números o símbolos
    if re.fullmatch(r"[\d\s\W]+", t):
        return True

    if _looks_like_toc(t):
        return True

    return False


def cargar_pdf(ruta_pdf: str) -> list[Document]:
    doc = fitz.open(ruta_pdf)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ": ",
            " "
        ],
    )

    documentos: list[Document] = []
    chunk_id = 0

    for page_num, pagina in enumerate(doc):
        raw = pagina.get_text()
        cleaned = _clean(raw)

        if not cleaned:
            continue

        chunks = splitter.split_text(cleaned)

        for chunk in chunks:
            chunk = chunk.strip()

            if _is_noise(chunk):
                continue

            documentos.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": ruta_pdf,
                        "page": page_num + 1,
                        "chunk_id": chunk_id,
                    },
                )
            )
            chunk_id += 1

    return documentos


if __name__ == "__main__":
    ruta_pdf = "data/Doc chatbot.pdf"
    documentos = cargar_pdf(ruta_pdf)

    print(f"✅ Total de fragmentos: {len(documentos)}\n")
    for i, d in enumerate(documentos[:5]):
        print(f"[{i}] metadata={d.metadata}")
        print(d.page_content[:300], "\n")