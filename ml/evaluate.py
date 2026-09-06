import os
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from ml.model import DermDiagModel
from ml.dataset import get_train_val_loaders
from ml.class_mapping import HAM10000_CLASSES, CLASS_NAMES
from backend.config import settings


def evaluate(batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DermDiagModel.load_from_checkpoint(settings.MODEL_PATH, device=str(device))
    _, val_loader = get_train_val_loaders(settings.HAM10000_DATA_DIR, batch_size=batch_size)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Macro Precision: {prec:.4f}")
    print(f"Macro Recall: {rec:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("\nPer-class metrics:")
    for i, cls in enumerate(HAM10000_CLASSES):
        cls_prec = precision_score(all_labels, all_preds, average=None)[i]
        cls_rec = recall_score(all_labels, all_preds, average=None)[i]
        cls_f1 = f1_score(all_labels, all_preds, average=None)[i]
        print(f"  {cls} ({CLASS_NAMES[cls]}): P={cls_prec:.4f} R={cls_rec:.4f} F1={cls_f1:.4f}")
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    evaluate()
