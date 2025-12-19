import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image


# ----------------------------
# Dataset Loader
# ----------------------------
class FaceDatasetCNN(Dataset):
    def __init__(self, fake_dir, real_dir, transform=None):
        self.items = []

        for f in os.listdir(fake_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.items.append((os.path.join(fake_dir, f), 1))

        for f in os.listdir(real_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                self.items.append((os.path.join(real_dir, f), 0))

        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------------------
# Train CNN (MobileNetV2)
# ----------------------------
def train_cnn():

    fake_dir = "data/faces_fake"
    real_dir = "data/faces_real"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training MobileNetV2 CNN on:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = FaceDatasetCNN(fake_dir, real_dir, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Load pretrained MobileNetV2
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

        acc = correct / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f} Acc: {acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_mobilenetv2.pt")
    print("✓ MobileNetV2 CNN saved at models/cnn_mobilenetv2.pt")


if __name__ == "__main__":
    train_cnn()
