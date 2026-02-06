import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="session")
def client():
    os.environ["RAG_MODE"] = "baseline"
    return TestClient(app)
