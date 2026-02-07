from pathlib import Path
from backend.Pdf_loader import cargar_pdf
import pytest
from pathlib import Path

pdf_path = Path("data") / "Doc chatbot.pdf"

pytestmark = pytest.mark.skipif(
    not pdf_path.exists(),
    reason="PDF not available (skipping integration-style test)."
)






def _extract_text(chunk):
    # LangChain Document: .page_content
    if hasattr(chunk, 'page_content'):
        return chunk.page_content
    # dict típico: {'text': ...} o {'page_content': ...}
    if isinstance(chunk, dict):
        return chunk.get('page_content') or chunk.get('text') or chunk.get('content')
    # tuple/list típico: (text, meta)
    if isinstance(chunk, (tuple, list)) and len(chunk) > 0:
        if isinstance(chunk[0], str):
            return chunk[0]
    # fallback: convertir a str
    return str(chunk)

def test_pdf_loader_carga_archivo():
    pdf_path = Path('data') / 'Doc chatbot.pdf'
    assert pdf_path.exists(), 'El PDF de riesgos no existe'

    chunks = cargar_pdf(str(pdf_path))

    assert chunks is not None
    assert len(chunks) > 0

    texts = [_extract_text(c) for c in chunks]
    assert all(isinstance(t, str) for t in texts)
    assert all(len(t.strip()) > 0 for t in texts)
