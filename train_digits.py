"""
Main training script for the CAPTCHA character classifier CNN.

Each time you run this script:
1. It regenerates a fresh batch of 10,000 character images (calls generate_char_dataset.py)
2. It loads the previous model checkpoint if one exists (cumulative learning across runs)
3. It trains for a few epochs on the new data
4. It saves the updated model + appends this run's accuracy to a history log
5. It updates a chart (accuracy_history.png) showing accuracy improving over multiple runs

Just run:  python3 train_captcha_cnn.py
"""

import os
import csv
import shutil
import subprocess
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt

# ---- Config ----
DATA_DIR = "./data"
IMAGE_SIZE = 32
CHECKPOINT_PATH = "captcha_cnn.pth"
HISTORY_CSV = "accuracy_history.csv"
HISTORY_CHART = "accuracy_history.png"
GENERATE_SCRIPT = "generate_new_charac.py"

EPOCHS_PER_RUN = 5      # epochs trained THIS run (on top of previous cumulative training)
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# ---- Step 0: Clean up old images, then regenerate a fresh batch ----
if os.path.exists(DATA_DIR):
    print(f">>> Removing old dataset at '{DATA_DIR}'...")
    shutil.rmtree(DATA_DIR)

print(">>> Generating a fresh batch of 10,000 training images...")
subprocess.run(["python3", GENERATE_SCRIPT], check=True)

# ---- Step 1: Setup device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Step 2: Load dataset ----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Loaded {len(full_dataset)} images across {num_classes} classes")

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, val_data = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# ---- Step 3: Define CNN architecture ----
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

model = CharCNN(num_classes, IMAGE_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

run_number = 1

# ---- Step 4: Load previous checkpoint if it exists (cumulative learning) ----
if os.path.exists(CHECKPOINT_PATH):
    print(f">>> Found existing checkpoint '{CHECKPOINT_PATH}', resuming training...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Sanity check: class list must match (folder structure shouldn't change between runs)
    if checkpoint["class_names"] != class_names:
        print("WARNING: class names in checkpoint don't match current dataset classes!")
        print("Training a fresh model instead to avoid mismatched layers.")
    else:
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
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1}/{EPOCHS_PER_RUN}, Loss: {avg_loss:.4f}")

# ---- Step 6: Validation accuracy for this run ----
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_accuracy = 100 * correct / total
print(f">>> Run #{run_number} validation accuracy: {val_accuracy:.2f}%")

# ---- Step 7: Save checkpoint (model + optimizer + metadata) ----
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "class_names": class_names,
    "image_size": IMAGE_SIZE,
    "run_number": run_number,
}, CHECKPOINT_PATH)
print(f">>> Checkpoint saved to {CHECKPOINT_PATH}")

# ---- Step 8: Append this run's result to the history log ----
file_exists = os.path.exists(HISTORY_CSV)
with open(HISTORY_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["run_number", "timestamp", "avg_loss", "val_accuracy"])
    writer.writerow([run_number, datetime.datetime.now().isoformat(timespec="seconds"),
                      f"{avg_loss:.4f}", f"{val_accuracy:.2f}"])

# ---- Step 9: Plot accuracy history across all runs ----
runs, accuracies = [], []
with open(HISTORY_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        runs.append(int(row["run_number"]))
        accuracies.append(float(row["val_accuracy"]))

plt.figure(figsize=(8, 5))
plt.plot(runs, accuracies, marker="o")
plt.xlabel("Training Run #")
plt.ylabel("Validation Accuracy (%)")
plt.title("CAPTCHA Character CNN — Accuracy Across Training Runs")
plt.grid(True)
plt.tight_layout()
plt.savefig(HISTORY_CHART)
print(f">>> Accuracy history chart updated: {HISTORY_CHART}")