
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert "model_ready" in response.json()


def test_rejects_non_image_content_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "document.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415
