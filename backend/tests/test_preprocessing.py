import torch
from PIL import Image
from backend.core.preprocessing import preprocess_image


def test_preprocess_output_shape():
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_different_sizes():
    for size in [(100, 100), (500, 300), (224, 224)]:
        img = Image.new("RGB", size, (128, 128, 128))
        tensor = preprocess_image(img)
        assert tensor.shape == (1, 3, 224, 224)
