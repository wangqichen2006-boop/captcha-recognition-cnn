"""
Full end-to-end pipeline: given a real CAPTCHA image with 5 characters,
1. Uses the LOCALIZATION model (localization_cnn.pth) to find where each
   of the 5 characters is.
2. Crops out each of those 5 regions.
3. Feeds each crop to your existing RECOGNITION model (captcha_cnn.pth,
   the one you already trained many runs on) to identify the character.
4. Combines the 5 results into the final predicted CAPTCHA string.

Usage:
    python3 predict_pipeline.py my_captcha.png
"""

import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

LOCALIZATION_CHECKPOINT = "localization_cnn.pth"
RECOGNITION_CHECKPOINT = "captcha_cnn.pth"


# ---- Localization model definition (must match train_localization.py) ----
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
        self.fc2 = nn.Linear(512, captcha_length * 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


# ---- Recognition model definition (must match main_train.py) ----
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


if len(sys.argv) < 2:
    print("Usage: python3 predict_pipeline.py <full_captcha_image_path>")
    sys.exit(1)
image_path = sys.argv[1]


def label_to_char(label):
    """Your recognition model's class folders are named like 'upper_N',
    'lower_y', 'digit_4' (to avoid case-collision issues), not a bare
    single character. This converts a class label back into the actual
    single character it represents."""
    for prefix in ("upper_", "lower_", "digit_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label  # already a bare single character, no prefix to strip


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Load localization model ----
loc_checkpoint = torch.load(LOCALIZATION_CHECKPOINT, map_location=device)
captcha_length = loc_checkpoint["captcha_length"]
img_h = loc_checkpoint["img_height"]
img_w = loc_checkpoint["img_width"]

loc_model = LocalizationCNN(captcha_length, img_h, img_w).to(device)
loc_model.load_state_dict(loc_checkpoint["model_state_dict"])
loc_model.eval()
print(f"Loaded localization model (trained {loc_checkpoint.get('run_number', '?')} run(s))")

# ---- Load recognition model ----
rec_checkpoint = torch.load(RECOGNITION_CHECKPOINT, map_location=device)
class_names = rec_checkpoint["class_names"]
rec_image_size = rec_checkpoint["image_size"]

rec_model = CharCNN(num_classes=len(class_names), image_size=rec_image_size).to(device)
rec_model.load_state_dict(rec_checkpoint["model_state_dict"])
rec_model.eval()
print(f"Loaded recognition model (trained {rec_checkpoint.get('run_number', '?')} run(s))")

# ---- Step 1: Load and preprocess the full CAPTCHA image for localization ----
full_img = Image.open(image_path).convert("L")

loc_transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
])
loc_tensor = loc_transform(full_img).unsqueeze(0).to(device)

if loc_tensor.mean() > 0.5:
    loc_tensor_for_model = 1.0 - loc_tensor
else:
    loc_tensor_for_model = loc_tensor

# ---- Step 2: Predict the 5 bounding boxes ----
with torch.no_grad():
    pred_boxes_norm = loc_model(loc_tensor_for_model).view(captcha_length, 4).cpu()

# Convert normalized boxes back to pixel coordinates in the ORIGINAL image size
orig_w, orig_h = full_img.size
boxes = []
for (l, t, r, b) in pred_boxes_norm.tolist():
    left = int(l * orig_w)
    top = int(t * orig_h)
    right = int(r * orig_w)
    bottom = int(b * orig_h)
    boxes.append((left, top, right, bottom))

print("\nPredicted character locations:")
for i, box in enumerate(boxes):
    print(f"  Char {i+1}: box = {box}")

# ---- Step 3: Crop each region and feed to the recognition model ----
rec_transform = transforms.Compose([
    transforms.Resize((rec_image_size, rec_image_size)),
    transforms.ToTensor(),
])

# DEBUG: save each cropped region as its own file so you can visually check
# whether the localization model is cropping the right area. If a crop looks
# wrong (half a character, wrong character, empty, etc.), the problem is in
# localization, not recognition — even if the recognition model reports high
# confidence on that bad crop.
debug_dir = "debug_crops"
os.makedirs(debug_dir, exist_ok=True)

result = ""
print("\nRecognizing each cropped character:")
for i, box in enumerate(boxes):
    # guard against degenerate/inverted boxes from an undertrained localization model
    left, top, right, bottom = box
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1

    char_crop = full_img.crop((left, top, right, bottom))
    char_crop.save(os.path.join(debug_dir, f"position_{i+1}_box_{left}_{top}_{right}_{bottom}.png"))

    char_tensor = rec_transform(char_crop).unsqueeze(0).to(device)

    if char_tensor.mean() > 0.5:
        char_tensor = 1.0 - char_tensor

    with torch.no_grad():
        output = rec_model(char_tensor)
        probs = torch.softmax(output, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        conf = probs[0][idx].item() * 100

    char = label_to_char(class_names[idx])
    result += char
    print(f"  Position {i+1}: '{char}' ({conf:.1f}% confidence)")

print(f"\n>>> Final predicted CAPTCHA: {result}")
print(f">>> Debug crops saved to '{debug_dir}/' — open them to check if localization cropped the right regions.")