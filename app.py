"""
Flask backend for the CAPTCHA demo webpage.

Serves:
  GET  /                 -> the HTML page
  POST /api/generate     -> generates a fresh CAPTCHA, runs it through your
                             localization model + recognition model, and
                             returns the image (base64) + prediction + per-
                             character confidence as JSON.

Run with:
    python3 app.py
Then open in your browser:
    http://localhost:5000
"""

import base64
import io
import random

import torch
import torch.nn as nn
from flask import Flask, jsonify, send_from_directory
from torchvision import transforms

import generate_full_captcha as gen  # reuses your existing image generator

# ---- Config ----
LOCALIZATION_CHECKPOINT = "localization_cnn.pth"
RECOGNITION_CHECKPOINT = "captcha_cnn.pth"

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- Model definitions (must match the training scripts exactly) ----
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


def label_to_char(label):
    """Converts folder-style labels like 'upper_N' / 'lower_y' / 'digit_4'
    back into the actual single character."""
    for prefix in ("upper_", "lower_", "digit_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


# ---- Load both models ONCE when the server starts (not per-request — much faster) ----
print("Loading localization model...")
loc_checkpoint = torch.load(LOCALIZATION_CHECKPOINT, map_location=device)
captcha_length = loc_checkpoint["captcha_length"]
img_h = loc_checkpoint["img_height"]
img_w = loc_checkpoint["img_width"]

loc_model = LocalizationCNN(captcha_length, img_h, img_w).to(device)
loc_model.load_state_dict(loc_checkpoint["model_state_dict"])
loc_model.eval()
print(f"  Loaded (trained {loc_checkpoint.get('run_number', '?')} run(s))")

print("Loading recognition model...")
rec_checkpoint = torch.load(RECOGNITION_CHECKPOINT, map_location=device)
class_names = rec_checkpoint["class_names"]
rec_image_size = rec_checkpoint["image_size"]

rec_model = CharCNN(num_classes=len(class_names), image_size=rec_image_size).to(device)
rec_model.load_state_dict(rec_checkpoint["model_state_dict"])
rec_model.eval()
print(f"  Loaded (trained {rec_checkpoint.get('run_number', '?')} run(s))")

loc_transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.ToTensor(),
])
rec_transform = transforms.Compose([
    transforms.Resize((rec_image_size, rec_image_size)),
    transforms.ToTensor(),
])


def run_pipeline(pil_image):
    """Runs the full localization -> recognition pipeline on a PIL image.
    Returns (predicted_string, list_of_{char, confidence, box})."""
    img_gray = pil_image.convert("L")

    # ---- Localization ----
    loc_tensor = loc_transform(img_gray).unsqueeze(0).to(device)
    if loc_tensor.mean() > 0.5:
        loc_tensor = 1.0 - loc_tensor

    with torch.no_grad():
        pred_boxes_norm = loc_model(loc_tensor).view(captcha_length, 4).cpu()

    orig_w, orig_h = img_gray.size
    boxes = []
    for (l, t, r, b) in pred_boxes_norm.tolist():
        left = int(l * orig_w)
        top = int(t * orig_h)
        right = int(r * orig_w)
        bottom = int(b * orig_h)
        if right <= left:
            right = left + 1
        if bottom <= top:
            bottom = top + 1
        boxes.append((left, top, right, bottom))

    # ---- Recognition ----
    results = []
    predicted_string = ""
    for box in boxes:
        char_crop = img_gray.crop(box)
        char_tensor = rec_transform(char_crop).unsqueeze(0).to(device)
        if char_tensor.mean() > 0.5:
            char_tensor = 1.0 - char_tensor

        with torch.no_grad():
            output = rec_model(char_tensor)
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            conf = probs[0][idx].item() * 100

        char = label_to_char(class_names[idx])
        predicted_string += char
        results.append({"char": char, "confidence": round(conf, 2), "box": box})

    return predicted_string, results


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    # 1. Generate a brand-new random CAPTCHA using your existing generator
    true_label = "".join(random.choice(gen.CHARACTERS) for _ in range(gen.CAPTCHA_LENGTH))
    image, true_boxes = gen.generate_captcha_image(true_label)

    # 2. Run your two trained models on it
    predicted_string, char_results = run_pipeline(image)

    # 3. Encode the image as base64 so it can be embedded directly in the HTML response
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "image_base64": image_base64,
        "true_label": true_label,          # the real answer (since we generated it ourselves)
        "predicted": predicted_string,
        "characters": char_results,
        "correct": predicted_string == true_label,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
