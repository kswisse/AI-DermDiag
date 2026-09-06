import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/dermdiag_model.pt")
    HAM10000_DATA_DIR: str = os.getenv("HAM10000_DATA_DIR", "./data")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: list = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))


settings = Settings()
