import io
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.validation import validate_image, ValidationError
from backend.core.inference import predict
from backend.metadata.class_info import get_class_info
from ml.class_mapping import HAM10000_CLASSES, CLASS_NAMES

router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    return {"status": "ok", "service": "AI-DermDiag"}


@router.get("/classes")
async def classes():
    return {
        "classes": [get_class_info(code) for code in HAM10000_CLASSES]
    }


@router.get("/info")
async def info():
    return {
        "name": "AI-DermDiag",
        "description": "AI-assisted skin lesion screening with explainable AI",
        "model": "EfficientNet-B0",
        "dataset": "HAM10000",
        "num_classes": len(HAM10000_CLASSES),
        "classes": CLASS_NAMES,
    }


def image_to_base64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post("/predict")
async def predict_endpoint(image: UploadFile = File(...)):
    try:
        file_bytes = await image.read()
        pil_image = validate_image(file_bytes, image.filename)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to process uploaded file.")
    try:
        result = predict(pil_image)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, detail="Model not found. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    class_info = get_class_info(result["predicted_class"])
    return {
        "predicted_class": result["predicted_class"],
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "top_predictions": result["top_predictions"],
        "original_image": f"data:image/png;base64,{image_to_base64(result['original_image'])}",
        "heatmap_image": f"data:image/png;base64,{image_to_base64(result['heatmap_image'])}",
        "overlay_image": f"data:image/png;base64,{image_to_base64(result['overlay_image'])}",
        "class_info": class_info,
        "disclaimer": (
            "This is an AI-assisted screening result and does not establish a medical diagnosis. "
            "Please consult a qualified healthcare professional for definitive evaluation."
        ),
        "model_version": "efficientnet-b0-ham10000-v1",
    }
