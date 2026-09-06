# AI-DermDiag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, runnable AI-assisted skin lesion screening web application using HAM10000 dataset, EfficientNet-B0, Grad-CAM explainability, with React frontend and FastAPI backend.

**Architecture:** Python FastAPI backend serves a React frontend and provides `/api/predict` endpoint. PyTorch EfficientNet-B0 model trained on HAM10000 performs 7-class classification. Grad-CAM generates explainability heatmaps. The entire app can run as a single process (backend serves frontend static files).

**Tech Stack:** Python 3.10+, FastAPI, PyTorch, torchvision, Grad-CAM (custom implementation), React 18, Vite, TailwindCSS, pytest, JavaScript

---

## Global Constraints

- HAM10000 dataset — 7 classes: akiec, bcc, bkl, df, mel, nv, vasc
- Model: EfficientNet-B0, input 224×224, 7-class output
- No fake predictions, no fabricated metrics, no mock Grad-CAM
- Medical disclaimer mandatory on all result screens
- CORS configured for development
- Environment variables via `.env` / `.env.example`
- All tests must pass before claiming completion

---

## File Structure

```
AI Dermdiag/
├── backend/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings & env vars
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # API routes
│   │   └── dependencies.py        # Shared dependencies
│   ├── core/
│   │   ├── __init__.py
│   │   ├── inference.py           # Model loading & prediction
│   │   ├── gradcam.py             # Grad-CAM implementation
│   │   ├── preprocessing.py       # Image preprocessing
│   │   └── validation.py          # File validation
│   ├── metadata/
│   │   ├── __init__.py
│   │   └── class_info.py          # HAM10000 class metadata
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_health.py
│       ├── test_validation.py
│       ├── test_preprocessing.py
│       ├── test_inference.py
│       ├── test_gradcam.py
│       └── test_predict_endpoint.py
├── ml/
│   ├── __init__.py
│   ├── dataset.py                 # HAM10000 dataset loader
│   ├── model.py                   # EfficientNet-B0 wrapper
│   ├── train.py                   # Training script
│   ├── evaluate.py                # Evaluation script
│   └── class_mapping.py           # Class mapping utilities
├── models/                        # Trained model weights (gitignored)
│   └── .gitkeep
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── App.css
│       ├── components/
│       │   ├── Header.jsx
│       │   ├── ImageUpload.jsx
│       │   ├── AnalysisResult.jsx
│       │   ├── GradCAMViewer.jsx
│       │   ├── ProbabilityChart.jsx
│       │   ├── Disclaimer.jsx
│       │   ├── ProjectInfo.jsx
│       │   └── ModelPerformance.jsx
│       └── utils/
│           └── api.js
├── tests/
│   └── e2e/
│       └── test_full_flow.py      # End-to-end integration test
├── .env.example
├── .gitignore
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `backend/__init__.py`
- Create: `ml/__init__.py`
- Create: `models/.gitkeep`

**Interfaces:** None (setup only)

- [ ] **Step 1: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*$py.class
.env
models/*.pt
models/*.pth
models/*.onnx
node_modules/
frontend/dist/
*.egg-info/
.pytest_cache/
.DS_Store
```

- [ ] **Step 2: Create `.env.example`**

```
MODEL_PATH=models/dermdiag_model.pt
HAM10000_DATA_DIR=./data
HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
MAX_UPLOAD_SIZE_MB=10
```

- [ ] **Step 3: Create `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
python-dotenv==1.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
tqdm>=4.65.0
pytest==8.3.0
httpx==0.27.0
```

- [ ] **Step 4: Create directory structure**

```powershell
mkdir backend\api, backend\core, backend\metadata, backend\tests, ml, models, frontend\src\components, frontend\src\utils, tests\e2e
```

- [ ] **Step 5: Create `__init__.py` files**

Create empty `__init__.py` in: `backend/`, `backend/api/`, `backend/core/`, `backend/metadata/`, `backend/tests/`, `ml/`, `tests/`, `tests/e2e/`

- [ ] **Step 6: Create `models/.gitkeep`**

- [ ] **Step 7: Install Python dependencies**

```powershell
pip install -r requirements.txt
```

- [ ] **Step 8: Commit**

```bash
git init
git add .
git commit -m "feat: project scaffolding and dependencies"
```

---

### Task 2: Class Mapping & Metadata

**Files:**
- Create: `ml/class_mapping.py`
- Create: `backend/metadata/class_info.py`

**Interfaces:**
- Consumes: HAM10000 class definitions
- Produces: `CLASS_NAMES`, `CLASS_DESCRIPTIONS`, `get_class_info(code)`, `idx_to_class`, `class_to_idx`

- [ ] **Step 1: Write `ml/class_mapping.py`**

```python
HAM10000_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

CLASS_TO_IDX = {cls: i for i, cls in enumerate(HAM10000_CLASSES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(HAM10000_CLASSES)}
```

- [ ] **Step 2: Write `backend/metadata/class_info.py`**

```python
from ml.class_mapping import HAM10000_CLASSES, CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS

CLASS_INFO = {
    "akiec": {
        "code": "akiec",
        "name": "Actinic Keratoses",
        "description": "Pre-cancerous skin growths caused by sun damage. Also known as solar keratoses.",
        "risk": "Potentially pre-malignant. Professional evaluation recommended.",
    },
    "bcc": {
        "code": "bcc",
        "name": "Basal Cell Carcinoma",
        "description": "The most common type of skin cancer, arising from basal cells in the epidermis.",
        "risk": "Malignant. Professional dermatological evaluation strongly recommended.",
    },
    "bkl": {
        "code": "bkl",
        "name": "Benign Keratosis",
        "description": "Common benign skin growths including seborrheic keratoses and lichen planus-like keratoses.",
        "risk": "Generally benign. Monitor for changes.",
    },
    "df": {
        "code": "df",
        "name": "Dermatofibroma",
        "description": "A common benign skin nodule, often found on the lower legs.",
        "risk": "Benign. No treatment usually needed unless symptomatic.",
    },
    "mel": {
        "code": "mel",
        "name": "Melanoma",
        "description": "A malignant melanocytic skin tumor. The most dangerous form of skin cancer.",
        "risk": "Malignant. Urgent professional dermatological evaluation strongly recommended.",
    },
    "nv": {
        "code": "nv",
        "name": "Melanocytic Nevi",
        "description": "Common moles. Benign proliferation of melanocytes.",
        "risk": "Generally benign. Monitor for changes in size, shape, or color.",
    },
    "vasc": {
        "code": "vasc",
        "name": "Vascular Lesions",
        "description": "Skin lesions containing blood vessels, including angiomas and pyogenic granulomas.",
        "risk": "Usually benign. Professional evaluation if new or changing.",
    },
}

def get_class_info(code: str) -> dict:
    return CLASS_INFO.get(code, {
        "code": code,
        "name": CLASS_NAMES.get(code, code),
        "description": "No description available.",
        "risk": "Consult a healthcare professional.",
    })

def get_display_name(code: str) -> str:
    return CLASS_NAMES.get(code, code)

def get_risk_level(code: str) -> str:
    info = CLASS_INFO.get(code, {})
    return info.get("risk", "Unknown")
```

- [ ] **Step 3: Run test**

```powershell
python -c "from ml.class_mapping import HAM10000_CLASSES; from backend.metadata.class_info import get_class_info; print(get_class_info('mel'))"
```

Expected: dict with melanoma info

- [ ] **Step 4: Commit**

---

### Task 3: ML Model Definition

**Files:**
- Create: `ml/model.py`

**Interfaces:**
- Consumes: `ml.class_mapping.HAM10000_CLASSES`
- Produces: `DermDiagModel` class with `forward(x)`, `get_last_conv_layer()`

- [ ] **Step 1: Write `ml/model.py`**

```python
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
```

- [ ] **Step 2: Run verification**

```powershell
python -c "from ml.model import DermDiagModel; m = DermDiagModel(pretrained=False); print('Last conv:', m.get_last_conv_layer()); import torch; x = torch.randn(1, 3, 224, 224); print('Output shape:', m(x).shape)"
```

Expected: `Output shape: torch.Size([1, 7])`

- [ ] **Step 3: Commit**

---

### Task 4: Dataset Loader

**Files:**
- Create: `ml/dataset.py`

**Interfaces:**
- Consumes: HAM10000 CSV metadata + images directory
- Produces: `HAM10000Dataset`, `get_transforms()`, `get_train_val_loaders()`

- [ ] **Step 1: Write `ml/dataset.py`**

```python
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from ml.class_mapping import CLASS_TO_IDX

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image_id"] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = CLASS_TO_IDX[row["dx"]]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

def get_train_val_loaders(data_dir, batch_size=32, val_split=0.2):
    metadata_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    img_dir = os.path.join(data_dir, "images")
    df = pd.read_csv(metadata_path)
    df = df[df["image_id"].apply(lambda x: os.path.exists(os.path.join(img_dir, x + ".jpg")))]
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df["dx"], random_state=42)
    train_ds = HAM10000Dataset(train_df, img_dir, get_transforms("train"))
    val_ds = HAM10000Dataset(val_df, img_dir, get_transforms("val"))
    class_counts = df["dx"].value_counts().sort_index()
    sample_weights = [1.0 / class_counts[row["dx"]] for _, row in train_df.iterrows()]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
```

- [ ] **Step 2: Commit**

---

### Task 5: Grad-CAM Implementation

**Files:**
- Create: `backend/core/gradcam.py`

**Interfaces:**
- Consumes: PyTorch model, input tensor
- Produces: `GradCAM` class with `generate(input_tensor)` returning heatmap overlay

- [ ] **Step 1: Write `backend/core/gradcam.py`**

```python
import torch
import numpy as np
from PIL import Image
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam, size=(224, 224), mode="bilinear", align_corners=False
        )
        return cam.squeeze().cpu().numpy()

    @staticmethod
    def overlay_heatmap(original_image, heatmap, alpha=0.5):
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image.resize((224, 224)))
        elif isinstance(original_image, torch.Tensor):
            original_image = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
            original_image = (original_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_resized * 255), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = (original_image * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
        return Image.fromarray(overlay)

    @staticmethod
    def heatmap_to_image(heatmap):
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(heatmap_resized * 255), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        return Image.fromarray(heatmap_colored)
```

- [ ] **Step 2: Commit**

---

### Task 6: Image Preprocessing & Validation

**Files:**
- Create: `backend/core/preprocessing.py`
- Create: `backend/core/validation.py`

**Interfaces:**
- Consumes: PIL Image, file bytes
- Produces: torch tensor, validation result

- [ ] **Step 1: Write `backend/core/preprocessing.py`**

```python
import torch
from torchvision import transforms
from PIL import Image

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    tensor = INFERENCE_TRANSFORM(image)
    return tensor.unsqueeze(0)
```

- [ ] **Step 2: Write `backend/core/validation.py`**

```python
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
        raise ValidationError(f"Unsupported file format '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
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
```

- [ ] **Step 3: Commit**

---

### Task 7: Inference Engine

**Files:**
- Create: `backend/core/inference.py`

**Interfaces:**
- Consumes: PIL Image, trained model
- Produces: prediction dict with class, confidence, top-k, heatmap

- [ ] **Step 1: Write `backend/core/inference.py`**

```python
import torch
import numpy as np
from PIL import Image
from ml.model import DermDiagModel
from ml.class_mapping import HAM10000_CLASSES, CLASS_NAMES, IDX_TO_CLASS
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
```

- [ ] **Step 2: Write `backend/config.py`**

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/dermdiag_model.pt")
    HAM10000_DATA_DIR: str = os.getenv("HAM10000_DATA_DIR", "./data")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

settings = Settings()
```

- [ ] **Step 3: Commit**

---

### Task 8: FastAPI Backend

**Files:**
- Create: `backend/main.py`
- Create: `backend/api/routes.py`

**Interfaces:**
- Consumes: inference engine, validation, preprocessing
- Produces: `/api/health`, `/api/predict`, `/api/info`, `/api/classes`

- [ ] **Step 1: Write `backend/api/routes.py`**

```python
import io
import base64
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core.validation import validate_image, ValidationError
from backend.core.inference import predict
from backend.metadata.class_info import CLASS_INFO, get_class_info
from ml.class_mapping import HAM10000_CLASSES, CLASS_NAMES

router = APIRouter(prefix="/api")

@router.get("/health")
async def health():
    return {"status": "ok", "service": "AI-DermDiag"}

@router.get("/classes")
async def classes():
    return {
        "classes": [
            get_class_info(code) for code in HAM10000_CLASSES
        ]
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
        raise HTTPException(status_code=503, detail="Model not found. Please train the model first.")
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
        "disclaimer": "This is an AI-assisted screening result and does not establish a medical diagnosis. Please consult a qualified healthcare professional for definitive evaluation.",
        "model_version": "efficientnet-b0-ham10000-v1",
    }
```

- [ ] **Step 2: Write `backend/main.py`**

```python
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import settings
from backend.api.routes import router

app = FastAPI(title="AI-DermDiag", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=True)
```

- [ ] **Step 3: Commit**

---

### Task 9: Training Pipeline

**Files:**
- Create: `ml/train.py`
- Create: `ml/evaluate.py`

**Interfaces:**
- Consumes: HAM10000 dataset, DermDiagModel
- Produces: trained model weights, class mapping, evaluation metrics

- [ ] **Step 1: Write `ml/train.py`**

```python
import os
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from ml.model import DermDiagModel
from ml.dataset import get_train_val_loaders
from ml.class_mapping import HAM10000_CLASSES
from backend.config import settings

def train(epochs=25, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    data_dir = settings.HAM10000_DATA_DIR
    train_loader, val_loader = get_train_val_loaders(data_dir, batch_size=batch_size)
    model = DermDiagModel(num_classes=len(HAM10000_CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    os.makedirs("models", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += images.size(0)
        train_acc = train_correct / train_total
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / val_total
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": HAM10000_CLASSES,
                "val_acc": val_acc,
            }, settings.MODEL_PATH)
            print(f"  Saved best model (val_acc={val_acc:.4f})")
    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
```

- [ ] **Step 2: Write `ml/evaluate.py`**

```python
import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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
    return {"accuracy": acc, "f1": f1, "precision": prec, "recall": rec, "confusion_matrix": cm.tolist()}

if __name__ == "__main__":
    evaluate()
```

- [ ] **Step 3: Commit**

---

### Task 10: Frontend — React + Vite Setup

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/main.jsx`
- Create: `frontend/src/App.jsx`
- Create: `frontend/src/App.css`

**Interfaces:** None (setup only)

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "ai-dermdiag-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.3.0",
    "vite": "^5.4.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
```

- [ ] **Step 2: Create `frontend/vite.config.js`**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000'
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true
  }
})
```

- [ ] **Step 3: Create `frontend/tailwind.config.js`**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        medical: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        }
      }
    }
  },
  plugins: []
}
```

- [ ] **Step 4: Create `frontend/postcss.config.js`**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  }
}
```

- [ ] **Step 5: Create `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI-DermDiag</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.jsx"></script>
</body>
</html>
```

- [ ] **Step 6: Create `frontend/src/main.jsx`**

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

- [ ] **Step 7: Create `frontend/src/App.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background-color: #f8fafc;
  margin: 0;
}
```

- [ ] **Step 8: Create `frontend/src/App.jsx`**

```jsx
import React, { useState } from 'react'
import Header from './components/Header'
import ImageUpload from './components/ImageUpload'
import AnalysisResult from './components/AnalysisResult'
import Disclaimer from './components/Disclaimer'
import ProjectInfo from './components/ProjectInfo'
import { analyzeImage } from './utils/api'

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [view, setView] = useState('upload')

  const handleAnalyze = async (file) => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await analyzeImage(file)
      setResult(data)
      setView('result')
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setResult(null)
    setError(null)
    setView('upload')
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <Header />
      <main className="max-w-5xl mx-auto px-4 py-8">
        {view === 'upload' && (
          <>
            <ImageUpload onAnalyze={handleAnalyze} loading={loading} />
            {error && (
              <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}
          </>
        )}
        {view === 'result' && result && (
          <AnalysisResult result={result} onReset={handleReset} />
        )}
        <Disclaimer />
      </main>
      <ProjectInfo />
    </div>
  )
}
```

- [ ] **Step 9: Commit**

---

### Task 11: Frontend — Components

**Files:**
- Create: `frontend/src/components/Header.jsx`
- Create: `frontend/src/components/ImageUpload.jsx`
- Create: `frontend/src/components/AnalysisResult.jsx`
- Create: `frontend/src/components/GradCAMViewer.jsx`
- Create: `frontend/src/components/ProbabilityChart.jsx`
- Create: `frontend/src/components/Disclaimer.jsx`
- Create: `frontend/src/components/ProjectInfo.jsx`
- Create: `frontend/src/utils/api.js`

- [ ] **Step 1: Write `frontend/src/utils/api.js`**

```javascript
const API_BASE = '/api'

export async function analyzeImage(file) {
  const formData = new FormData()
  formData.append('image', file)
  const response = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    throw new Error(data?.detail || `Server error: ${response.status}`)
  }
  return response.json()
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`)
  return response.json()
}

export async function fetchClasses() {
  const response = await fetch(`${API_BASE}/classes`)
  return response.json()
}
```

- [ ] **Step 2: Write `Header.jsx`**

```jsx
import React from 'react'

export default function Header() {
  return (
    <header className="bg-white border-b border-slate-200 shadow-sm">
      <div className="max-w-5xl mx-auto px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-medical-600 rounded-lg flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-800">AI-DermDiag</h1>
            <p className="text-sm text-slate-500">AI-assisted skin lesion screening with explainable AI</p>
          </div>
        </div>
        <div className="flex gap-2 mt-3">
          {['HAM10000', 'EfficientNet-B0', 'Grad-CAM', '7-class'].map(tag => (
            <span key={tag} className="px-2 py-1 bg-medical-50 text-medical-700 text-xs font-medium rounded-full">
              {tag}
            </span>
          ))}
        </div>
      </div>
    </header>
  )
}
```

- [ ] **Step 3: Write `ImageUpload.jsx`**

```jsx
import React, { useState, useRef } from 'react'

export default function ImageUpload({ onAnalyze, loading }) {
  const [preview, setPreview] = useState(null)
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleFile = (f) => {
    if (!f) return
    if (!f.type.match(/^image\/(jpeg|jpg|png|webp)$/)) {
      alert('Please upload a JPG, PNG, or WEBP image.')
      return
    }
    if (f.size > 10 * 1024 * 1024) {
      alert('File too large. Maximum size is 10MB.')
      return
    }
    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files[0])
  }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8">
      <div
        className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors cursor-pointer
          ${dragActive ? 'border-medical-500 bg-medical-50' : 'border-slate-300 hover:border-medical-400'}`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png,image/webp"
          className="hidden"
          onChange={(e) => handleFile(e.target.files[0])}
        />
        {preview ? (
          <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg shadow-sm" />
        ) : (
          <div>
            <svg className="w-16 h-16 mx-auto text-slate-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <p className="text-slate-600 font-medium">Upload a skin lesion image</p>
            <p className="text-slate-400 text-sm mt-1">JPG, PNG — max 10MB</p>
          </div>
        )}
      </div>
      {file && (
        <button
          onClick={() => onAnalyze(file)}
          disabled={loading}
          className="mt-6 w-full py-3 bg-medical-600 hover:bg-medical-700 disabled:bg-slate-400 text-white font-medium rounded-lg transition-colors"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Analyzing image...
            </span>
          ) : (
            'Analyze image'
          )}
        </button>
      )}
      {loading && (
        <div className="mt-4 text-center text-sm text-slate-500 space-y-1">
          <p>Running EfficientNet-B0 inference</p>
          <p>Generating explainability map</p>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Write `AnalysisResult.jsx`**

```jsx
import React, { useState } from 'react'
import GradCAMViewer from './GradCAMViewer'
import ProbabilityChart from './ProbabilityChart'

export default function AnalysisResult({ result, onReset }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-slate-800">AI Screening Result</h2>
        <button onClick={onReset} className="px-4 py-2 text-sm text-medical-600 hover:text-medical-700 font-medium">
          New Analysis
        </button>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
        <div className="text-center mb-4">
          <p className="text-sm text-slate-500 uppercase tracking-wide">Predicted Class</p>
          <p className="text-3xl font-bold text-slate-800 mt-1">{result.class_name}</p>
          <p className="text-lg text-medical-600 font-semibold mt-1">
            Confidence {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>

        {result.class_info && (
          <div className="bg-slate-50 rounded-lg p-4 mb-4">
            <p className="text-sm text-slate-600">{result.class_info.description}</p>
            {result.class_info.risk && (
              <p className="text-sm text-amber-700 mt-2 font-medium">{result.class_info.risk}</p>
            )}
          </div>
        )}
      </div>

      <ProbabilityChart predictions={result.top_predictions} />

      <GradCAMViewer
        original={result.original_image}
        heatmap={result.heatmap_image}
        overlay={result.overlay_image}
      />

      <div className="bg-amber-50 border border-amber-200 rounded-xl p-5">
        <p className="text-sm text-amber-800">
          <strong>Important:</strong> {result.disclaimer}
        </p>
      </div>
    </div>
  )
}
```

- [ ] **Step 5: Write `GradCAMViewer.jsx`**

```jsx
import React, { useState } from 'react'

const VIEWS = [
  { key: 'original', label: 'Original' },
  { key: 'heatmap', label: 'Heatmap' },
  { key: 'overlay', label: 'Overlay' },
]

export default function GradCAMViewer({ original, heatmap, overlay }) {
  const [active, setActive] = useState('overlay')
  const images = { original, heatmap, overlay }

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-2">Explainability</h3>
      <p className="text-sm text-slate-500 mb-4">
        The Grad-CAM visualization highlights image regions that contributed most strongly to the model's prediction.
      </p>
      <div className="flex gap-2 mb-4">
        {VIEWS.map(v => (
          <button
            key={v.key}
            onClick={() => setActive(v.key)}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors
              ${active === v.key ? 'bg-medical-600 text-white' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}
          >
            {v.label}
          </button>
        ))}
      </div>
      <div className="flex justify-center">
        <img
          src={images[active]}
          alt={active}
          className="max-h-96 rounded-lg shadow-sm"
        />
      </div>
    </div>
  )
}
```

- [ ] **Step 6: Write `ProbabilityChart.jsx`**

```jsx
import React from 'react'

export default function ProbabilityChart({ predictions }) {
  const maxProb = Math.max(...predictions.map(p => p.probability))
  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <h3 className="text-lg font-semibold text-slate-800 mb-4">Top Predictions</h3>
      <div className="space-y-3">
        {predictions.map((pred, i) => (
          <div key={pred.class} className="flex items-center gap-3">
            <span className="text-sm font-medium text-slate-500 w-5">{i + 1}.</span>
            <span className="text-sm font-medium text-slate-700 w-36 truncate">{pred.name}</span>
            <div className="flex-1 bg-slate-100 rounded-full h-4 overflow-hidden">
              <div
                className="h-full bg-medical-500 rounded-full transition-all duration-500"
                style={{ width: `${(pred.probability / maxProb) * 100}%` }}
              />
            </div>
            <span className="text-sm font-mono text-slate-600 w-16 text-right">
              {(pred.probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 7: Write `Disclaimer.jsx`**

```jsx
import React from 'react'

export default function Disclaimer() {
  return (
    <div className="mt-8 bg-slate-100 border border-slate-200 rounded-xl p-5">
      <p className="text-sm text-slate-600 leading-relaxed">
        <strong>Important:</strong> AI-DermDiag is a research and screening prototype. Its output is not a
        definitive medical diagnosis and should not replace evaluation by a qualified healthcare professional.
        HAM10000 consists of dermatoscopic images and does not automatically represent smartphone-camera conditions.
        Dataset performance does not equal clinical performance. Grad-CAM is an interpretability aid, not proof
        of clinical reasoning. Professional evaluation is required for suspicious lesions.
      </p>
    </div>
  )
}
```

- [ ] **Step 8: Write `ProjectInfo.jsx`**

```jsx
import React from 'react'

export default function ProjectInfo() {
  return (
    <footer className="bg-white border-t border-slate-200 mt-12">
      <div className="max-w-5xl mx-auto px-4 py-8">
        <h2 className="text-xl font-bold text-slate-800 mb-4">About AI-DermDiag</h2>
        <div className="grid md:grid-cols-2 gap-6 text-sm text-slate-600">
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Problem</h3>
            <p>Skin cancer is one of the most common cancers worldwide. Early detection significantly improves outcomes, but access to dermatological expertise is limited in many regions.</p>
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Solution</h3>
            <p>AI-DermDiag provides AI-assisted screening of skin lesion images using deep learning, offering preliminary classifications with explainable visualizations.</p>
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Dataset</h3>
            <p>Trained on HAM10000 — Human Against Machine with 10,000 training images. Contains 10,015 dermatoscopic images across 7 diagnostic categories.</p>
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-2">Technology</h3>
            <p>EfficientNet-B0 for classification, Grad-CAM for explainability. Built with PyTorch, FastAPI, and React.</p>
          </div>
        </div>
        <div className="mt-6 pt-4 border-t border-slate-200 text-sm text-slate-500">
          <p><strong>Classes:</strong> Actinic Keratoses (akiec), Basal Cell Carcinoma (bcc), Benign Keratosis (bkl), Dermatofibroma (df), Melanoma (mel), Melanocytic Nevi (nv), Vascular Lesions (vasc)</p>
          <p className="mt-2"><strong>Limitations:</strong> This prototype is not clinically validated. Dermatoscopic images differ from smartphone photos. Performance may vary across demographics. Not a standalone diagnostic tool.</p>
        </div>
        <p className="mt-4 text-xs text-slate-400">Developed by Develop4Life — Nguyễn Đinh Trọng Khang, Nguyễn Đinh Bích Khuê</p>
      </div>
    </footer>
  )
}
```

- [ ] **Step 9: Install frontend dependencies**

```powershell
cd frontend; npm install
```

- [ ] **Step 10: Build frontend**

```powershell
cd frontend; npm run build
```

- [ ] **Step 11: Commit**

---

### Task 12: Backend Tests

**Files:**
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_health.py`
- Create: `backend/tests/test_validation.py`
- Create: `backend/tests/test_preprocessing.py`
- Create: `backend/tests/test_inference.py`
- Create: `backend/tests/test_gradcam.py`
- Create: `backend/tests/test_predict_endpoint.py`

- [ ] **Step 1: Write `conftest.py`**

```python
import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_image_bytes():
    from PIL import Image
    import io
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
```

- [ ] **Step 2: Write `test_health.py`**

```python
def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "AI-DermDiag"
```

- [ ] **Step 3: Write `test_validation.py`**

```python
import io
from PIL import Image
from backend.core.validation import validate_image, ValidationError
import pytest

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
```

- [ ] **Step 4: Write `test_preprocessing.py`**

```python
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
```

- [ ] **Step 5: Write `test_gradcam.py`**

```python
import torch
from ml.model import DermDiagModel
from backend.core.gradcam import GradCAM
from PIL import Image

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
    import numpy as np
    heatmap = np.random.rand(224, 224).astype(np.float32)
    overlay = GradCAM.overlay_heatmap(img, heatmap)
    assert overlay.size == (224, 224)
```

- [ ] **Step 6: Write `test_inference.py`**

```python
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from backend.core.inference import predict
from ml.model import DermDiagModel

@pytest.fixture
def mock_model():
    model = DermDiagModel(pretrained=False)
    model.eval()
    return model

@patch("backend.core.inference._model", None)
@patch("backend.core.inference.load_model")
def test_predict_returns_valid_result(mock_load_model, mock_model):
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
```

- [ ] **Step 7: Write `test_predict_endpoint.py`**

```python
import io
from PIL import Image
from unittest.mock import patch
import pytest

@pytest.fixture
def sample_image_file():
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return ("test.png", buf, "image/png")

def test_predict_endpoint_no_model(client):
    response = client.post("/api/predict", files={"image": ("test.png", b"fake", "image/png")})
    assert response.status_code in [400, 503]

def test_predict_endpoint_invalid_format(client):
    response = client.post("/api/predict", files={"image": ("test.bmp", b"fake", "image/bmp")})
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]

def test_predict_endpoint_empty_file(client):
    response = client.post("/api/predict", files={"image": ("test.png", b"", "image/png")})
    assert response.status_code == 400
```

- [ ] **Step 8: Run all backend tests**

```powershell
python -m pytest backend/tests/ -v
```

- [ ] **Step 9: Commit**

---

### Task 13: Docker Configuration

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN cd frontend && npm install && npm run build

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models/dermdiag_model.pt
      - HAM10000_DATA_DIR=/app/data
```

- [ ] **Step 3: Commit**

---

### Task 14: README & Documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# AI-DermDiag

AI-assisted skin lesion screening using HAM10000 dataset with EfficientNet-B0 and Grad-CAM explainability.

## Architecture

```
AI-DermDiag/
├── backend/     FastAPI server with inference, Grad-CAM, validation
├── frontend/    React + Vite + TailwindCSS
├── ml/          Dataset loader, model, training, evaluation
├── models/      Trained model weights
└── tests/       Backend and E2E tests
```

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, PyTorch
- **Frontend:** React 18, Vite, TailwindCSS
- **ML:** EfficientNet-B0, Grad-CAM, HAM10000
- **Tests:** pytest, httpx

## Setup

### 1. Clone and install

```bash
git clone <repo>
cd AI-Dermdiag
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Download HAM10000

Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Extract to `data/` directory:
```
data/
├── HAM10000_metadata.csv
└── images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ...
```

### 3. Train the model

```bash
python -m ml.train --epochs 25 --batch-size 32
```

### 4. Evaluate

```bash
python -m ml.evaluate
```

### 5. Run the app

```bash
python -m backend.main
```

Open http://localhost:8000

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /api/health | Health check |
| GET | /api/classes | List all classes |
| GET | /api/info | Model information |
| POST | /api/predict | Upload image for prediction |

## HAM10000 Classes

| Code | Name | Description |
|------|------|-------------|
| akiec | Actinic Keratoses | Pre-cancerous sun damage growths |
| bcc | Basal Cell Carcinoma | Most common skin cancer |
| bkl | Benign Keratosis | Common benign growths |
| df | Dermatofibroma | Benign skin nodule |
| mel | Melanoma | Malignant skin tumor |
| nv | Melanocytic Nevi | Common moles |
| vasc | Vascular Lesions | Blood vessel lesions |

## Medical Disclaimer

AI-DermDiag is a research and screening prototype. Its output is not a definitive medical diagnosis and should not replace evaluation by a qualified healthcare professional.

## Docker

```bash
docker-compose up --build
```

## Developed by

Develop4Life — Nguyễn Đinh Trọng Khang, Nguyễn Đinh Bích Khuê
```

- [ ] **Step 2: Commit**

---

### Task 15: End-to-End Verification

**Files:** None (verification only)

- [ ] **Step 1: Run all backend tests**

```powershell
python -m pytest backend/tests/ -v --tb=short
```

- [ ] **Step 2: Verify backend starts**

```powershell
python -c "from backend.main import app; print('Backend imports OK')"
```

- [ ] **Step 3: Verify frontend builds**

```powershell
cd frontend; npm run build
```

- [ ] **Step 4: Verify ML model loads (without trained weights, test architecture)**

```powershell
python -c "from ml.model import DermDiagModel; m = DermDiagModel(pretrained=False); print('Model architecture OK')"
```

- [ ] **Step 5: Verify Grad-CAM works**

```powershell
python -c "
from ml.model import DermDiagModel
from backend.core.gradcam import GradCAM
import torch
m = DermDiagModel(pretrained=False)
gc = GradCAM(m, m.get_last_conv_layer())
x = torch.randn(1,3,224,224)
h = gc.generate(x)
gc.remove_hooks()
print('Grad-CAM OK, shape:', h.shape)
"
```

- [ ] **Step 6: Commit final state**

```bash
git add -A
git commit -m "feat: complete AI-DermDiag implementation"
```
