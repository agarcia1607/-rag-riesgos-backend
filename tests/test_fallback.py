import os

def test_fallback_llm_to_baseline_returns_contract(client):
    # Forzamos el modo LLM
    os.environ['RAG_MODE'] = 'llm'

    r = client.post('/preguntar', json={'texto': '¿Qué riesgos cubre la póliza?'})
    assert r.status_code == 200

    data = r.json()
    assert 'respuesta' in data
    assert 'fuentes' in data
    assert isinstance(data['respuesta'], str)
    assert isinstance(data['fuentes'], list)

    # Limpieza (para no contaminar otros tests)
    os.environ['RAG_MODE'] = 'baseline'
