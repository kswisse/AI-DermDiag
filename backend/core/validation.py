from PIL import Image
import io

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_MB = 10


class ValidationError(Exception):
    pass


def validate_image(file_bytes: bytes, filename: str) -> Image.Image:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file format '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    if len(file_bytes) == 0:
        raise ValidationError("Empty file uploaded.")
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValidationError(f"File too large ({size_mb:.1f}MB). Maximum: {MAX_FILE_SIZE_MB}MB.")
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except Exception:
        raise ValidationError("Corrupted or unreadable image file.")
    return image
