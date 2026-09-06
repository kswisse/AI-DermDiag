import torch
import numpy as np
from PIL import Image
from ml.model import DermDiagModel
from backend.core.gradcam import GradCAM


def test_gradcam_generates_heatmap():
    model = DermDiagModel(pretrained=False)
    model.eval()
    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)
    input_tensor = torch.randn(1, 3, 224, 224)
    heatmap = grad_cam.generate(input_tensor)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0
    grad_cam.remove_hooks()


def test_heatmap_overlay():
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    heatmap = np.random.rand(224, 224).astype(np.float32)
    overlay = GradCAM.overlay_heatmap(img, heatmap)
    assert overlay.size == (224, 224)
