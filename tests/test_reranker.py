from backend.retrievers.reranker import Reranker


def test_reranker_orders_relevant_doc_first():
    reranker = Reranker()

    query = "¿Cuándo puede sustituirse la escolta satelital?"
    docs = [
        {"text": "La póliza cubre transporte terrestre nacional."},
        {"text": "La escolta satelital puede sustituirse por monitoreo satelital vía mobile en ciertos casos."},
        {"text": "El deducible aplica según la mercancía transportada."},
    ]

    results = reranker.rerank(query, docs, top_k=2)

    assert len(results) == 2
    assert "sustituirse por monitoreo satelital" in results[0]["text"].lower()
    assert "rerank_score" in results[0]
