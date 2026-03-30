from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ---------------- BASIC ----------------
def test_app_starts_and_responds():
    response = client.get("/docs")
    assert response.status_code == 200


# ---------------- UPLOAD ----------------
def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_accepts_pdf(monkeypatch):
    # 🔥 mock ingestion (avoid real Qdrant call)
    async def mock_ingest(*args, **kwargs):
        return None

    monkeypatch.setattr("main.rag_pipeline.ingest", mock_ingest)

    response = client.post(
        "/upload",
        files={"file": ("test.pdf", b"%PDF-1.4 fake", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "doc_id" in data
    assert data["message"] == "File uploaded and indexed successfully"


# ---------------- QUERY ----------------
def test_query_success(monkeypatch):
    async def mock_run(*args, **kwargs):
        return {
            "source": "rag",
            "answer": "Test answer",
            "score": 0.9
        }

    monkeypatch.setattr("main.team.run", mock_run)

    response = client.post(
        "/query",
        json={
            "doc_id": "test-id",
            "query": "What is AI?"
        },
    )

    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "Test answer"
    assert data["source"] == "rag"
    assert data["score"] == 0.9


def test_query_failure(monkeypatch):
    async def mock_run(*args, **kwargs):
        raise Exception("fail")

    monkeypatch.setattr("main.team.run", mock_run)

    response = client.post(
        "/query",
        json={
            "doc_id": "test-id",
            "query": "fail case"
        },
    )

    assert response.status_code == 500