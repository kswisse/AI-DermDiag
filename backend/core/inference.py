import torch
import numpy as np
from PIL import Image
from ml.model import DermDiagModel
from ml.class_mapping import CLASS_NAMES, IDX_TO_CLASS
from backend.core.preprocessing import preprocess_image
from backend.core.gradcam import GradCAM
from backend.config import settings

_model = None
_device = None


def load_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = DermDiagModel.load_from_checkpoint(settings.MODEL_PATH, device=str(_device))
    _model.eval()
    return _model, _device


def predict(image: Image.Image, top_k: int = 7) -> dict:
    model, device = load_model()
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
    top_probs, top_indices = probs.topk(top_k)
    predicted_idx = top_indices[0].item()
    predicted_class = IDX_TO_CLASS[predicted_idx]
    confidence = top_probs[0].item()
    top_predictions = []
    for i in range(top_k):
        idx = top_indices[i].item()
        cls_code = IDX_TO_CLASS[idx]
        top_predictions.append({
            "class": cls_code,
            "name": CLASS_NAMES.get(cls_code, cls_code),
            "probability": round(probs[idx].item(), 4),
        })
    grad_cam = GradCAM(model, model.get_last_conv_layer())
    heatmap = grad_cam.generate(input_tensor, target_class=predicted_idx)
    grad_cam.remove_hooks()
    original_resized = image.resize((224, 224))
    heatmap_image = GradCAM.heatmap_to_image(heatmap)
    overlay_image = GradCAM.overlay_heatmap(original_resized, heatmap)
    return {
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES.get(predicted_class, predicted_class),
        "confidence": round(confidence, 4),
        "top_predictions": top_predictions,
        "original_image": original_resized,
        "heatmap_image": heatmap_image,
        "overlay_image": overlay_image,
    }
