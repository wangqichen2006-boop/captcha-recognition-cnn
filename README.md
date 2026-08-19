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
├── train_digits.py            # Trains the RECOGNITION model (captcha_cnn.pth)
├── predict_single_char.py     # Test the recognition model on one character image
│
├── generate_full_captcha.py   # Generates full 5-character CAPTCHA images + ground-truth
│                               # box coordinates (data_full/, boxes.json)
├── train_localization.py      # Trains the LOCALIZATION model (localization_cnn.pth)
│
├── predict_pipeline.py        # Full pipeline: localize -> crop -> recognize
├── run_manytimes.py           # Runs a training script N times in a row
│
├── app.py                     # Flask backend for the web demo
├── index.html                 # Web demo frontend (served by app.py)
│
└── README.md
```

> The two training scripts (`train_digits.py`, `train_localization.py`) are self-contained: each run deletes old generated images, generates a fresh 10,000-image batch, loads the previous model checkpoint if one exists, trains a few epochs, saves the checkpoint, and appends the run's accuracy to a CSV log + chart.

## Setup

1. Clone the repo and `cd` into it.
2. Install the dependencies listed above.
3. Confirm your GPU is visible to PyTorch:
   ```bash
   python3 -c "import torch; print(torch.cuda.is_available())"
   ```

## Script-by-script usage

### `generate_new_charac.py`
Generates 10,000 single-character images (62 classes: `0-9`, `A-Z`, `a-z`) into `data/`, one subfolder per class. Applies randomized distortions (rotation, shear, perspective, pixelation, grid lines, noise lines, blur) for visual variety. You normally don't run this directly — `train_digits.py` calls it automatically before every training run. Can be run standalone to preview generated data:
```bash
python3 generate_new_charac.py
```

### `train_digits.py`
Trains the **recognition** model — classifies a single cropped character image into one of 62 classes.
```bash
python3 train_digits.py
```
Produces/updates:
- `captcha_cnn.pth` — model checkpoint (cumulative across runs — each run resumes from the last)
- `accuracy_history.csv` / `accuracy_history.png` — accuracy log and chart

### `predict_single_char.py`
Loads `captcha_cnn.pth` and runs it on a single character image, printing the top-5 predicted characters with confidence.
```bash
python3 predict_single_char.py path/to/character.png
```

### `generate_full_captcha.py`
Generates 10,000 full 5-character CAPTCHA images into `data_full/`, plus a `boxes.json` file recording the ground-truth bounding box of each character (used to train the localization model). You normally don't run this directly — `train_localization.py` calls it automatically before every training run. Can be run standalone to preview generated data:
```bash
python3 generate_full_captcha.py
```

### `train_localization.py`
Trains the **localization** model — given a full CAPTCHA image, predicts the bounding box (left, top, right, bottom) of each of the 5 characters. This is a regression task (predicting coordinates), evaluated using IoU (Intersection over Union) against the ground-truth boxes.
```bash
python3 train_localization.py
```
Produces/updates:
- `localization_cnn.pth` — model checkpoint (cumulative across runs)
- `accuracy_history_localization.csv` / `accuracy_history_localization.png`

### `run_manytimes.py`
Convenience script that runs `train_digits.py` back-to-back N times, so you don't have to manually re-run the command yourself. Stops early if any run fails.
```bash
python3 run_manytimes.py 20
```
If no number is given, defaults to 5 runs. To use it with `train_localization.py` instead, edit the `TRAIN_SCRIPT` variable at the top of the file.

### `predict_pipeline.py`
The full end-to-end pipeline. Requires both `captcha_cnn.pth` and `localization_cnn.pth` to exist. Given a full CAPTCHA image:
1. Runs the localization model to find the 5 character bounding boxes.
2. Crops each region.
3. Runs the recognition model on each crop.
4. Combines the 5 results into the final predicted string.
```bash
python3 predict_pipeline.py path/to/captcha.png
```
Prints the predicted boxes, the recognized character + confidence at each position, and the final combined string. Also saves each cropped character to `debug_crops/` so you can visually confirm the localization model cropped the right regions.

### `app.py` + `index.html`
A Flask web demo. On startup, loads both trained models once. Serves `index.html` at `/`, and exposes a `POST /api/generate` endpoint that generates a brand-new random CAPTCHA, runs the full pipeline on it, and returns the image + prediction + per-character confidence as JSON. `index.html` is the frontend — a page with a button that calls this endpoint and renders the result, including a green/red correctness indicator and a confidence bar per character.
```bash
python3 app.py
```
Then open `http://localhost:5000` in a browser.

## How cumulative training works

Both `train_digits.py` and `train_localization.py` follow the same pattern:

1. Delete the old generated dataset and generate a fresh batch of images.
2. If a checkpoint file exists, load the saved model + optimizer state and resume from there; otherwise start from scratch.
3. Train for a fixed number of epochs.
4. Save the updated checkpoint (overwriting the old one) and append this run's accuracy to a history log.

This means you can safely re-run either training script any number of times — each run builds on the last, and the accuracy charts show progress across all runs, not just the most recent one.

## Notes

- Training data is 100% synthetically generated (random characters + fonts + distortions: rotation, shear, perspective, pixelation, grid lines, noise). No real CAPTCHA images are collected, scraped, or required.
- Model checkpoints (`*.pth`) and generated image folders (`data/`, `data_full/`) are excluded from version control via `.gitignore` — they're large and easily regenerated/retrained locally.
