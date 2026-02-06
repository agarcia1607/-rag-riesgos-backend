import os

EMPTY_MSG = 'No encontré fragmentos relevantes para esa consulta en el documento.'

def test_baseline_mode_contract(client):
    os.environ['RAG_MODE'] = 'baseline'
    r = client.post('/preguntar', json={'texto': 'riesgos cubiertos'})
    assert r.status_code == 200

    data = r.json()
    assert isinstance(data.get('respuesta'), str)
    assert isinstance(data.get('fuentes'), list)

def test_baseline_empty_response_when_no_evidence(client):
    os.environ['RAG_MODE'] = 'baseline'

    # Consulta deliberadamente rara para intentar forzar "sin evidencia"
    r = client.post('/preguntar', json={'texto': 'zzzxqv___consulta_totalmente_inexistente___123'})
    assert r.status_code == 200

    data = r.json()
    assert 'respuesta' in data and 'fuentes' in data

    # Si no hay evidencia, debe seguir el estándar
    if data['fuentes'] == []:
        assert data['respuesta'] == EMPTY_MSG
