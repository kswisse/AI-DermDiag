import pytest
from unittest.mock import patch
from PIL import Image
from ml.model import DermDiagModel


@pytest.fixture
def mock_model():
    model = DermDiagModel(pretrained=False)
    model.eval()
    return model


@patch("backend.core.inference._model", None)
@patch("backend.core.inference.load_model")
def test_predict_returns_valid_result(mock_load_model, mock_model):
    from backend.core.inference import predict

    mock_load_model.return_value = (mock_model, "cpu")
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    result = predict(img)
    assert "predicted_class" in result
    assert "confidence" in result
    assert "top_predictions" in result
    assert "heatmap_image" in result
    assert "overlay_image" in result
    assert len(result["top_predictions"]) == 7
    assert 0 <= result["confidence"] <= 1
    total_prob = sum(p["probability"] for p in result["top_predictions"])
    assert abs(total_prob - 1.0) < 0.01
