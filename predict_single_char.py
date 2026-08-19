"""
Test the trained CAPTCHA character CNN on a single character image.

Usage:
    python3 predict_single_char.py my_image.png
"""

import sys
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

CHECKPOINT_PATH = "captcha_cnn.pth"

# ---- Must match the architecture used in main_train.py exactly ----
class CharCNN(nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        reduced = image_size // 4
        self.fc1 = nn.Linear(64 * reduced * reduced, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---- Parse image path ----
if len(sys.argv) < 2:
    print("Usage: python3 predict_single_char.py <image_path>")
    sys.exit(1)
image_path = sys.argv[1]

# ---- Load model + metadata ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
class_names = checkpoint["class_names"]
image_size = checkpoint["image_size"]
run_number = checkpoint.get("run_number", "?")

model = CharCNN(num_classes=len(class_names), image_size=image_size).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded model (trained for {run_number} run(s)), {len(class_names)} classes")

# ---- Preprocess the image ----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])

img = Image.open(image_path)
img_tensor = transform(img).unsqueeze(0).to(device)

# Auto-fix color inversion: model expects white character on black background.
if img_tensor.mean() > 0.5:
    img_tensor = 1.0 - img_tensor
    print("(Detected light background — auto-inverted colors)")

# ---- Predict ----
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)

    top5_probs, top5_idx = torch.topk(probs, 5)

print(f"\nImage: {image_path}")
print("Top 5 predictions:")
for i in range(5):
    char = class_names[top5_idx[0][i].item()]
    conf = top5_probs[0][i].item() * 100
    print(f"  {i+1}. '{char}'  —  {conf:.2f}%")

best_char = class_names[top5_idx[0][0].item()]
best_conf = top5_probs[0][0].item() * 100
print(f"\n>>> Predicted character: '{best_char}' (confidence: {best_conf:.2f}%)")   