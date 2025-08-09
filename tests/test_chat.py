from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_chat():
    res = client.post("/chat", data={"message": "Hello"})
    assert res.status_code == 200
    assert "answer" in res.json()
