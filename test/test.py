# tests/test_app_basic.py

from fastapi.testclient import TestClient

def test_app_starts_and_responds():
    # Import must succeed (this catches path + case issues)
    import main  

    assert hasattr(main, "app"), "FastAPI app object not found"

    client = TestClient(main.app)

    # Simple request to ensure ASGI stack boots
    response = client.get("/docs")

    assert response.status_code == 200




def test_upload_rejects_non_pdf():
    import main
    client = TestClient(main.app)

    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]