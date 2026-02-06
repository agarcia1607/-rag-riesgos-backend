import os
import time

def test_stability_and_latency_baseline(client):
    os.environ['RAG_MODE'] = 'baseline'

    start = time.time()

    for _ in range(5):
        r = client.post('/preguntar', json={'texto': 'riesgos'})
        assert r.status_code == 200
        data = r.json()
        assert 'respuesta' in data and 'fuentes' in data

    elapsed = time.time() - start
    assert elapsed < 8
