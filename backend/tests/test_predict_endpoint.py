import io
from PIL import Image
import pytest


def test_predict_endpoint_no_model(client):
    response = client.post(
        "/api/predict",
        files={"image": ("test.png", b"fake image data", "image/png")},
    )
    assert response.status_code in [400, 500, 503]


def test_predict_endpoint_invalid_format(client):
    response = client.post(
        "/api/predict",
        files={"image": ("test.bmp", b"fake", "image/bmp")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_predict_endpoint_empty_file(client):
    response = client.post(
        "/api/predict",
        files={"image": ("test.png", b"", "image/png")},
    )
    assert response.status_code == 400
