from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "AI-DermDiag"}


def test_info():
    resp = client.get("/api/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "EfficientNet-B0"
    assert len(body["classes"]) > 0


def test_classes():
    resp = client.get("/api/classes")
    assert resp.status_code == 200
    assert len(resp.json()["classes"]) > 0
