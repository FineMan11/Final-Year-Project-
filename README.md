# FYP Repo — Component Detection for Product Assembly (Portfolio-Ready)

> **Repository generated for:** Rais Hamizan Bin Faridan — *Final Year Project* (FYP)

---

This document contains the full **repository skeleton** and file contents for a **portfolio-ready** GitHub repo (option B). The repository is designed to be polished for both academic submission and job/portfolio presentation.

> **What this canvas contains:**
>
> * A complete repo tree
> * Ready-to-paste `README.md` (polished and professional)
> * Script templates (`train.py`, `detect.py`, `dataset_prep.py`, Colab notebook markdown, etc.)
> * `requirements.txt`, `LICENSE` (MIT), `.gitignore`
> * Documentation files (`ROBoflow_integration.md`, `RESULTS.md`, `DEPLOY.md`)
> * Guidance for uploading to GitHub and producing a short demo video

---

## Repository tree (recommended)

```
component-detection-fyp/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ data/
│  ├─ raw/                # raw images (do not upload > Git LFS recommended)
│  ├─ annotations/        # YOLO txt / COCO json
│  └─ metadata.csv        # image, bbox, class summary
├─ models/
│  ├─ yolov5s_best.pt     # (optional: symlink or release attachment)
│  └─ yolov5x_best.pt
├─ notebooks/
│  └─ Colab_Train.md      # Colab-ready notebook (markdown)
├─ src/
│  ├─ train.py
│  ├─ detect.py
│  ├─ utils/
│  │  ├─ dataset_prep.py
│  │  └─ viz.py
│  └─ config/
│     ├─ yolov5s.yaml
│     └─ yolov5x.yaml
├─ docs/
│  ├─ ROBoflow_integration.md
│  ├─ DEPLOY.md
│  └─ RESULTS.md
├─ assets/
│  ├─ demo_gif.gif
│  ├─ sample1.jpg
│  └─ sample2.jpg
└─ badges.md
```

---

> **Note:** Add large model files or the dataset to GitHub Releases, or use Git LFS. Do not push the raw dataset directly to the repo unless you use LFS or the dataset is small.

---

# Files created for you (paste into your repo)

> The full contents of each file are included below. Copy each code block into the corresponding file in your Git repo.

---

## 1) `README.md` (polished / copy-paste)

````markdown
# Component Detection for Product Assembly in Manufacturing

**Author:** Rais Hamizan Bin Faridan (A21EE0282) — Universiti Teknologi Malaysia

## Overview
This repository contains the code, documentation, and assets for the Final Year Project (FYP): **Component Detection for Product Assembly in Manufacturing using Deep Learning (YOLOv5)**.

The project focuses on detecting three component classes:
- `Arduino Uno`
- `Motor Driver`
- `Servo Motor`

Two YOLOv5 variants were trained and evaluated: **YOLOv5s** (fast) and **YOLOv5x** (accurate). The project emphasizes a balance between accuracy and latency for real-time manufacturing usage.

---

## Quick links
- 📄 Report / Thesis: `FYP2 Report.pdf` (attach in repo root)  
- 🧪 Notebooks: `notebooks/Colab_Train.md`  
- 📦 Models: `models/` (download from Releases)  

---

## Repo structure
Refer to the project tree in the repository root for a detailed structure.

---

## Install (local)
```bash
# Create venv (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
````

> GPU users: install appropriate CUDA + PyTorch versions. See `requirements.txt`.

---

## Quick start — inference

```bash
# Download model to models/yolov5s_best.pt (or yolov5x_best.pt)
python src/detect.py --weights models/yolov5s_best.pt --source assets/sample1.jpg --conf 0.25
```

Outputs will be saved in `runs/detect/` with bounding box visualizations and logs.

---

## Training (Colab)

1. Open `notebooks/Colab_Train.md` and follow the instructions to mount Google Drive or use Roboflow dataset export.
2. Upload dataset or link Roboflow dataset.
3. Run the training cell for the chosen config (`yolov5s.yaml` or `yolov5x.yaml`).

---

## Results (high level)

* YOLOv5x (960 images): Arduino 97.1%, Motor Driver 98%, Servo 100%
* YOLOv5s (960 images): Arduino 95.2%, Motor Driver 93.9%, Servo 95.7%
* Latency (Roboflow test): YOLOv5s ≈ 83 ms (≈13 fps), YOLOv5x ≈ 333 ms (≈3 fps)

> See `docs/RESULTS.md` for full metrics, confusion matrices and loss plots.

---

## Deployment suggestions

* For real-time on-device inference, use the **YOLOv5s** model and optimize with ONNX + TensorRT (NVIDIA) or TFLite (Edge TPU).
* For higher accuracy where latency is less critical, use **YOLOv5x** with GPU-backed inference.

Detailed deployment steps: `docs/DEPLOY.md`.

---

## License

This repository is released under the MIT License. See `LICENSE`.

---

## Contact

Rais Hamizan Bin Faridan — [rhfaridan@graduate.utm.my](mailto:rhfaridan@graduate.utm.my)

```
```

---

## 2) `requirements.txt`

```text
# Minimum recommended packages
torch>=1.12.0
torchvision
opencv-python
numpy
pandas
matplotlib
ultralytics>=8.0.0
roboflow
pyyaml
flask
tqdm
Pillow
```

---

## 3) `.gitignore`

```
__pycache__/
*.pyc
.env
venv/
models/*.pt
data/raw/
*.ipynb_checkpoints
runs/
.DS_Store
```

---

## 4) `LICENSE` (MIT)

```text
MIT License

Copyright (c) 2025 Rais Hamizan Bin Faridan

Permission is hereby granted, free of charge, to any person obtaining a copy
... (standard MIT text)
```

---

## 5) `src/train.py` (template)

```python
"""
Training wrapper that uses Ultralytics YOLOv5 training API.
This is a template — modify paths and hyperparameters as needed.
"""
import argparse
from ultralytics import YOLO


def main(args):
    # example with ultralytics YOLO
    model = YOLO(args.cfg)  # if using ultralytics v8 style; adapt to your environment
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='src/config/yolov5s.yaml')
    parser.add_argument('--data', type=str, default='data/data.yaml')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()
    main(args)
```

---

## 6) `src/detect.py` (template)

```python
"""
Simple inference script using ultralytics or torch.hub-yolov5.
"""
import argparse
from ultralytics import YOLO


def run(weights, source, conf=0.25, imgsz=640):
    model = YOLO(weights)
    results = model.predict(source, conf=conf, imgsz=imgsz)
    # results are saved in runs/predict by default


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    run(args.weights, args.source, args.conf)
```

---

## 7) `src/utils/dataset_prep.py` (template)

```python
"""
Utilities to prepare dataset into YOLO format.
"""
import os
import json
import shutil
import pandas as pd

# Implement functions: parse_coco_to_yolo, split_train_val, sanity_checks

```

---

## 8) `notebooks/Colab_Train.md`

```markdown
# Colab Training Guide (steps)

1. Open Google Colab and connect a GPU runtime.
2. Mount your Google Drive: `from google.colab import drive; drive.mount('/content/drive')`
3. Clone this repository: `git clone <your-repo-url>`
4. `pip install -r requirements.txt`
5. Upload dataset (or use Roboflow export link)
6. Run training: `python src/train.py --cfg src/config/yolov5s.yaml --data data/data.yaml --epochs 200`

Notes: Use `nvidia-smi` to check GPU. If using Ultralytics v8+, adapt commands to `yolo task=detect` syntax.
```

---

## 9) `docs/ROBoflow_integration.md`

````markdown
# Roboflow Integration (short)

1. Create Roboflow project and upload annotated images.
2. Export dataset in YOLOv5 PyTorch format.
3. Download zip and extract into `data/` or link Roboflow directly in Colab via the API key.

Example (python):
```python
from roboflow import Roboflow
rf = Roboflow(api_key="<API_KEY>")
project = rf.workspace().project("component-detection")
dataset = project.version(1).download("yolov5")
````

````

---

## 10) `docs/RESULTS.md`

```markdown
# Results Summary

Include here the images of the loss curves, confusion matrices and explanation copied from the thesis. Use the following structure:

- Training details
- Per-class accuracy
- Confusion matrices
- Latency tests (device + Roboflow results)

(Attach PNGs exported from training runs into `docs/assets/` and reference them.)
````

---

## 11) `docs/DEPLOY.md`

```markdown
# Deployment Guide (short)

Options:
- Edge (Raspberry Pi): convert to TFLite/Edge optimized weights
- NVIDIA Jetson: export to ONNX -> TensorRT
- Cloud API: wrap `src/detect.py` with FastAPI and serve on a GPU instance

Example: ONNX export (ultralytics): `python export.py --weights models/yolov5s_best.pt --onnx` (adapt to your environment)
```

---

## 12) `badges.md` (optional)

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python:3.x](https://img.shields.io/badge/python-3.8%2B-blue.svg)]
```

---

# Next steps I can do for you (pick any):

1. Create the **actual files** and zip them so you can download and push to GitHub.
2. Generate a polished `README.md` with figures (I will embed base64 images or placeholders).
3. Produce a short `demo script` and `sample inference video` instructions.
4. Prepare a GitHub release message + release assets (models + dataset pointers).

---

If you want me to **generate the zip with files now**, tell me **"Generate zip"** and I will prepare the code files and a downloadable archive.

If you prefer to copy-paste files yourself, you can start by creating the repository with the structure above and pasting the files from this canvas.

---

*End of auto-generated repository blueprint.*
