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
git clone https://github.com/kswisse/AI-DermDiag.git
cd AI-DermDiag
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Download the model weights

The trained checkpoint is distributed as a GitHub release asset (not stored in git):

```bash
mkdir -p models
curl -L -o models/dermdiag_model.pt https://github.com/kswisse/AI-DermDiag/releases/download/model-v1/dermdiag_model.pt
```

Windows (PowerShell):

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest -Uri https://github.com/kswisse/AI-DermDiag/releases/download/model-v1/dermdiag_model.pt -OutFile models/dermdiag_model.pt
```

### 3. Download HAM10000

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

### 4. Train the model

```bash
python -m ml.train --epochs 25 --batch-size 32
```

(Only needed if you want to retrain; step 2 already gives you a working checkpoint.)

### 5. Evaluate

```bash
python -m ml.evaluate
```

### 6. Run the app

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
