from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.models import mobilenet_v2
import shutil
import tempfile

# Import AutoEncoder from train_ae
from src.train_ae import AutoEncoder

app = FastAPI(title="Deepfake Detection API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------
# Load Models (similar to detect.py)
# ----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- Load MobileNetV2 CNN ----
cnn = mobilenet_v2(weights=None)
cnn.classifier[1] = torch.nn.Linear(cnn.classifier[1].in_features, 2)
cnn.load_state_dict(torch.load("../models/cnn_mobilenetv2.pt", map_location=device))
cnn = cnn.to(device)
cnn.eval()

# ---- Load AutoEncoder ----
ae = AutoEncoder().to(device)
ae.load_state_dict(torch.load("../models/ae.pt", map_location=device))
ae.eval()

# Face detector
mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=False)

# Transforms
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

ae_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# ----------------------------------
# Detection Function (adapted from detect.py)
# ----------------------------------
def detect_image(img_path):
    img = Image.open(img_path).convert("RGB")

    # Face Detection
    face = mtcnn(img)
    if face is None:
        return {"error": "No face detected"}

    # Convert to PIL
    if isinstance(face, torch.Tensor):
        face = face.permute(1, 2, 0).cpu().numpy()
    face_pil = Image.fromarray(face.astype("uint8"))

    # CNN Prediction
    cnn_input = cnn_transform(face_pil).unsqueeze(0).to(device)
    cnn_out = cnn(cnn_input)
    cnn_prob_fake = torch.softmax(cnn_out, dim=1)[0][1].item()

    # AutoEncoder Reconstruction
    ae_input = ae_transform(face_pil).unsqueeze(0).to(device)
    ae_recon = ae(ae_input)
    ae_loss = torch.mean(torch.abs(ae_input - ae_recon)).item()
    ae_norm = min(ae_loss / 0.20, 1.0)

    # Combine Results
    final_score = (0.60 * cnn_prob_fake) + (0.40 * ae_norm)
    prediction = "FAKE" if final_score > 0.50 else "REAL"

    return {
        "cnn_prob_fake": round(cnn_prob_fake, 3),
        "ae_error": round(ae_loss, 5),
        "ae_norm": round(ae_norm, 3),
        "final_score": round(final_score, 3),
        "prediction": prediction
    }

# ----------------------------------
# API Endpoints
# ----------------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        result = detect_image(temp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(temp_path)

@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    # Placeholder for video detection
    # For now, return a message
    return {"message": "Video detection not implemented yet. Please upload an image."}

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}