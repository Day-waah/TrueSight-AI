import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
from torchvision.models import mobilenet_v2
from train_ae import AutoEncoder


# ------------------------------------------
# Load Models
# ------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---- Load MobileNetV2 CNN ----
cnn = mobilenet_v2(weights=None)
cnn.classifier[1] = torch.nn.Linear(cnn.classifier[1].in_features, 2)
cnn.load_state_dict(torch.load("models/cnn_mobilenetv2.pt", map_location=device))
cnn = cnn.to(device)
cnn.eval()

# ---- Load AutoEncoder ----
ae = AutoEncoder().to(device)
ae.load_state_dict(torch.load("models/ae.pt", map_location=device))
ae.eval()

# Face detector (160×160 output)
mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=False)

# CNN transform
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# AutoEncoder transform
ae_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])


# ------------------------------------------
# Detection Function
# ------------------------------------------
def detect_image(img_path):

    img = Image.open(img_path).convert("RGB")

    # ---- Face Detection ----
    face = mtcnn(img)

    if face is None:
        return {"error": "No face detected"}

    # MTCNN → Numpy → PIL
    if isinstance(face, torch.Tensor):
        face = face.permute(1, 2, 0).cpu().numpy()

    face_pil = Image.fromarray(face.astype("uint8"))

    # ---------------- CNN Prediction ----------------
    cnn_input = cnn_transform(face_pil).unsqueeze(0).to(device)
    cnn_out = cnn(cnn_input)
    cnn_prob_fake = torch.softmax(cnn_out, dim=1)[0][1].item()

    # ---------------- AutoEncoder Reconstruction ----------------
    ae_input = ae_transform(face_pil).unsqueeze(0).to(device)
    ae_recon = ae(ae_input)
    ae_loss = torch.mean(torch.abs(ae_input - ae_recon)).item()

    # Normalize AE error
    ae_norm = min(ae_loss / 0.20, 1.0)

    # ---------------- Combine Results ----------------
    final_score = (0.60 * cnn_prob_fake) + (0.40 * ae_norm)

    prediction = "FAKE" if final_score > 0.50 else "REAL"

    return {
        "cnn_prob_fake": round(cnn_prob_fake, 3),
        "ae_error": round(ae_loss, 5),
        "ae_norm": round(ae_norm, 3),
        "final_score": round(final_score, 3),
        "prediction": prediction
    }


# ------------------------------------------
# Run from Terminal
# ------------------------------------------
if __name__ == "__main__":
    path = input("Enter image path: ").strip().strip('"')
    if os.path.exists(path):
        print(detect_image(path))
    else:
        print("❌ File not found:", path)
