def test_preguntar_contract(client):
    r = client.post('/preguntar', json={'texto': '¿Qué riesgos cubre la póliza?'})
    assert r.status_code == 200

    data = r.json()
    assert 'respuesta' in data
    assert 'fuentes' in data
    assert isinstance(data['respuesta'], str)
    assert isinstance(data['fuentes'], list)
    assert all(isinstance(x, str) for x in data['fuentes'])
