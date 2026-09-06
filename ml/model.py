import torch
import torch.nn as nn
from torchvision import models
from ml.class_mapping import HAM10000_CLASSES


class DermDiagModel(nn.Module):
    def __init__(self, num_classes=len(HAM10000_CLASSES), pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        return self.backbone.features[-1]

    @staticmethod
    def load_from_checkpoint(path, device="cpu", num_classes=len(HAM10000_CLASSES)):
        model = DermDiagModel(num_classes=num_classes, pretrained=False)
        state_dict = torch.load(path, map_location=device, weights_only=True)
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
