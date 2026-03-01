from fastapi.testclient import TestClient
from main import app   # app is the FastAPI instance


def test_app_starts_and_responds():
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200


def test_upload_rejects_non_pdf():
    client = TestClient(app)
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400