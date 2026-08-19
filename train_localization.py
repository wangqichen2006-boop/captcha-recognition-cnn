






"""
Trains a CNN to LOCATE the 5 characters in a full CAPTCHA image
(predicts bounding box coordinates), without trying to recognize
what each character is — that job stays with your existing
single-character recognition model (captcha_cnn.pth).

Each run:
1. Deletes old data_full/ and regenerates 10,000 fresh full-CAPTCHA
   images + their ground-truth box coordinates (boxes.json)
2. Loads the previous localization checkpoint if one exists
3. Trains a few epochs (regression task, not classification)
4. Saves checkpoint + logs mean IoU (overlap accuracy) + updates chart

Just run:  python3 train_localization.py
"""

import os
import csv
import json
import shutil
import subprocess
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Config (must match generate_full_captcha.py) ----
DATA_DIR = "./data_full"
CAPTCHA_LENGTH = 5
IMG_HEIGHT = 40
IMG_WIDTH = 32 * CAPTCHA_LENGTH
CHECKPOINT_PATH = "localization_cnn.pth"
HISTORY_CSV = "accuracy_history_localization.csv"
HISTORY_CHART = "accuracy_history_localization.png"
GENERATE_SCRIPT = "generate_full_captcha.py"

EPOCHS_PER_RUN = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# ---- Step 0: Clean up and regenerate fresh data ----
if os.path.exists(DATA_DIR):
    print(f">>> Removing old dataset at '{DATA_DIR}'...")
    shutil.rmtree(DATA_DIR)

print(">>> Generating a fresh batch of 10,000 full CAPTCHA images (with box labels)...")
subprocess.run(["python3", GENERATE_SCRIPT], check=True)

# ---- Step 1: Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ---- Step 2: Dataset — reads boxes.json for ground-truth coordinates ----
class LocalizationDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        with open(os.path.join(data_dir, "boxes.json")) as f:
            self.metadata = json.load(f)
        self.filenames = list(self.metadata.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.data_dir, fname))
        img = self.transform(img)

        boxes = self.metadata[fname]["boxes"]  # list of 5 [left, top, right, bottom]
        # normalize coordinates to 0-1 range so the model's output scale is stable
        norm_boxes = []
        for (l, t, r, b) in boxes:
            norm_boxes.extend([l / IMG_WIDTH, t / IMG_HEIGHT, r / IMG_WIDTH, b / IMG_HEIGHT])
        target = torch.tensor(norm_boxes, dtype=torch.float32)  # shape: [20] (5 boxes x 4 coords)
        return img, target


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
])

full_dataset = LocalizationDataset(DATA_DIR, transform)
print(f"Loaded {len(full_dataset)} images with box labels")

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)


# ---- Step 3: Localization model (regression, not classification) ----
class LocalizationCNN(nn.Module):
    def __init__(self, captcha_length, img_h, img_w):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)

        reduced_h = img_h // 8
        reduced_w = img_w // 8
        self.flatten_size = 128 * reduced_h * reduced_w

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, captcha_length * 4)   # 4 coords per character
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # squashes output to 0-1, matching our normalized targets

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


model = LocalizationCNN(CAPTCHA_LENGTH, IMG_HEIGHT, IMG_WIDTH).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()   # regression loss, not cross-entropy

run_number = 1

# ---- Step 4: Load previous checkpoint if it exists ----
if os.path.exists(CHECKPOINT_PATH):
    print(f">>> Found existing checkpoint '{CHECKPOINT_PATH}', resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    run_number = checkpoint.get("run_number", 0) + 1
else:
    print(">>> No existing checkpoint found, starting fresh.")

print(f">>> This is training run #{run_number}")

# ---- Step 5: Training loop ----
for epoch in range(EPOCHS_PER_RUN):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1}/{EPOCHS_PER_RUN}, Loss: {avg_loss:.6f}")


# ---- Step 6: Validation — mean IoU (Intersection over Union) across all boxes ----
def box_iou(box_a, box_b):
    """box_a, box_b: tensors of shape [..., 4] = (left, top, right, bottom), normalized 0-1."""
    left = torch.max(box_a[..., 0], box_b[..., 0])
    top = torch.max(box_a[..., 1], box_b[..., 1])
    right = torch.min(box_a[..., 2], box_b[..., 2])
    bottom = torch.min(box_a[..., 3], box_b[..., 3])

    inter_w = (right - left).clamp(min=0)
    inter_h = (bottom - top).clamp(min=0)
    inter_area = inter_w * inter_h

    area_a = (box_a[..., 2] - box_a[..., 0]).clamp(min=0) * (box_a[..., 3] - box_a[..., 1]).clamp(min=0)
    area_b = (box_b[..., 2] - box_b[..., 0]).clamp(min=0) * (box_b[..., 3] - box_b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter_area

    return inter_area / union.clamp(min=1e-6)


model.eval()
all_ious = []
with torch.no_grad():
    for images, targets in val_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)

        # reshape [batch, 20] -> [batch, 5, 4] so we can compare box-by-box
        pred_boxes = outputs.view(-1, CAPTCHA_LENGTH, 4)
        true_boxes = targets.view(-1, CAPTCHA_LENGTH, 4)

        ious = box_iou(pred_boxes, true_boxes)  # shape [batch, 5]
        all_ious.append(ious)

mean_iou = torch.cat(all_ious).mean().item()
# treat "IoU > 0.5" as a "correctly located" character, a common detection convention
localization_accuracy = (torch.cat(all_ious) > 0.5).float().mean().item() * 100

print(f">>> Run #{run_number} — Mean IoU: {mean_iou:.3f}, "
      f"Localization accuracy (IoU>0.5): {localization_accuracy:.2f}%")

# ---- Step 7: Save checkpoint ----
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "run_number": run_number,
    "captcha_length": CAPTCHA_LENGTH,
    "img_height": IMG_HEIGHT,
    "img_width": IMG_WIDTH,
}, CHECKPOINT_PATH)
print(f">>> Checkpoint saved to {CHECKPOINT_PATH}")

# ---- Step 8: Append to history log ----
file_exists = os.path.exists(HISTORY_CSV)
with open(HISTORY_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["run_number", "timestamp", "avg_loss", "mean_iou", "localization_accuracy"])
    writer.writerow([run_number, datetime.datetime.now().isoformat(timespec="seconds"),
                      f"{avg_loss:.6f}", f"{mean_iou:.3f}", f"{localization_accuracy:.2f}"])

# ---- Step 9: Plot accuracy history ----
runs, ious, accs = [], [], []
with open(HISTORY_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        runs.append(int(row["run_number"]))
        ious.append(float(row["mean_iou"]))
        accs.append(float(row["localization_accuracy"]))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(runs, accs, marker="o", color="tab:blue", label="Localization accuracy (IoU>0.5)")
ax1.set_xlabel("Training Run #")
ax1.set_ylabel("Localization Accuracy (%)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(runs, ious, marker="s", color="tab:orange", label="Mean IoU")
ax2.set_ylabel("Mean IoU", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")

plt.title("Character Localization CNN — Accuracy Across Training Runs")
fig.tight_layout()
plt.savefig(HISTORY_CHART)
print(f">>> Accuracy history chart updated: {HISTORY_CHART}")
