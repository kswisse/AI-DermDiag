import io
import pytest
from PIL import Image
from backend.core.validation import validate_image, ValidationError


def test_valid_png():
    img = Image.new("RGB", (100, 100), "red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = validate_image(buf.getvalue(), "test.png")
    assert result.size == (100, 100)


def test_valid_jpg():
    img = Image.new("RGB", (100, 100), "blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    result = validate_image(buf.getvalue(), "test.jpg")
    assert result.size == (100, 100)


def test_unsupported_format():
    with pytest.raises(ValidationError, match="Unsupported"):
        validate_image(b"fake", "test.bmp")


def test_empty_file():
    with pytest.raises(ValidationError, match="Empty"):
        validate_image(b"", "test.png")


def test_corrupted_image():
    with pytest.raises(ValidationError, match="Corrupted"):
        validate_image(b"not an image at all", "test.png")
