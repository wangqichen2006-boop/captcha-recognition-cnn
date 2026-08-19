# CAPTCHA Character Recognition

A two-stage deep learning pipeline that reads text-based CAPTCHAs:

1. **Localization model** — finds the bounding box of each character in a full CAPTCHA image.
2. **Recognition model** — classifies each cropped character (0–9, A–Z, a–z, 62 classes).

Both models are trained on synthetically generated CAPTCHA images (no real-world CAPTCHA data is used or required). Training is cumulative — each run regenerates a fresh batch of training images and continues training from the last saved checkpoint, so accuracy improves across repeated runs.

A small Flask web demo is included that generates a CAPTCHA, runs it through both models, and displays the predicted characters with per-character confidence.

## Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA support (training will fall back to CPU, but is much slower)
- PyTorch with CUDA support (`torch`, `torchvision`)
- `matplotlib`, `Pillow`, `numpy`
- `flask` (only needed for the web demo)

```bash
pip install torch torchvision matplotlib pillow numpy flask --break-system-packages
```

Recommended: install a few extra font packages so generated training characters have more visual variety.

```bash
sudo apt install fonts-dejavu-core fonts-liberation fonts-freefont-ttf fonts-noto-core -y
```

## Project structure

```
.
├── generate_new_charac.py     # Generates single-character training images (data/)
├── train_captcha_cnn.py       # Trains the RECOGNITION model (captcha_cnn.pth)
├── predict_single_char.py     # Test the recognition model on one character image
│
├── generate_full_captcha.py   # Generates full 5-character CAPTCHA images + ground-truth
│                               # box coordinates (data_full/, boxes.json)
├── train_localization.py      # Trains the LOCALIZATION model (localization_cnn.pth)
│
├── predict_pipeline.py        # Full pipeline: localize -> crop -> recognize
├── run_multiple_times.py      # Runs a training script N times in a row
│
├── app.py                     # Flask backend for the web demo
├── index.html                 # Web demo frontend
│
└── README.md
```

> Both training scripts are self-contained: each run deletes old generated images, generates a fresh 10,000-image batch, loads the previous model checkpoint if one exists, trains a few epochs, saves the checkpoint, and appends the run's accuracy to a CSV log + chart.

## Setup

1. Clone the repo and `cd` into it.
2. Install the dependencies listed above.
3. Confirm your GPU is visible to PyTorch:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

## Usage

### 1. Train the recognition model

```bash
python3 train_captcha_cnn.py
```

Produces/updates:
- `captcha_cnn.pth` — the model checkpoint (cumulative across runs)
- `accuracy_history.csv` / `accuracy_history.png` — accuracy log and chart

Run it repeatedly (or use the batch runner below) to keep improving accuracy:

```bash
python3 run_multiple_times.py 20
```

Test it on a single character image:

```bash
python3 predict_single_char.py path/to/character.png
```

### 2. Train the localization model

```bash
python3 train_localization.py
```

Produces/updates:
- `localization_cnn.pth` — the model checkpoint (cumulative across runs)
- `accuracy_history_localization.csv` / `accuracy_history_localization.png`

### 3. Run the full pipeline on a CAPTCHA image

Requires both `captcha_cnn.pth` and `localization_cnn.pth` to exist.

```bash
python3 predict_pipeline.py path/to/captcha.png
```

Outputs the predicted bounding boxes, the recognized character + confidence at each position, and the final combined string. Debug crops of each detected character are saved to `debug_crops/` for visual inspection.

### 4. Run the web demo

```bash
python3 app.py
```

Then open `http://localhost:5000` in a browser. Click the button to generate a new CAPTCHA and see both models run on it live, with per-character confidence.

## How cumulative training works

Every training script follows the same pattern:

1. Delete the old generated dataset and generate a fresh batch of images.
2. If a checkpoint file exists, load the saved model + optimizer state and resume from there; otherwise start from scratch.
3. Train for a fixed number of epochs.
4. Save the updated checkpoint (overwriting the old one) and append this run's accuracy to a history log.

This means you can safely re-run a training script any number of times — each run builds on the last, and the accuracy charts show progress across all runs, not just the most recent one.

## Notes

- Training data is 100% synthetically generated (random characters + fonts + distortions: rotation, shear, perspective, pixelation, grid lines, noise). No real CAPTCHA images are collected, scraped, or required.
- Model checkpoints (`*.pth`) and generated image folders (`data/`, `data_full/`) are excluded from version control via `.gitignore` — they're large and easily regenerated/retrained locally.
