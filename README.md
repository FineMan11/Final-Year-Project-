# Component Detection for Product Assembly Using YOLOv5
Automated detection of Arduino Uno, Motor Driver, and Servo Motor for manufacturing assembly verification.

![Demo](assets/demo.gif)

## Table of Contents
- Overview
- Features
- Dataset
- Training
- Results
- Installation
- Inference
- Deployment
- Project Structure
- License

## Overview
This project focuses on detecting 3 electronic components used in product assembly:
- Arduino Uno
- Motor Driver
- Servo Motor

The goal is to improve manufacturing quality control by automating component identification using YOLOv5.

## Features
- Trained YOLOv5s and YOLOv5x on 960 annotated images
- Real-time detection support (webcam)
- High accuracy (up to 100% on Servo Motor)
- Lightweight model available for fast deployment
- API ready (Flask) for integration into apps

## Dataset
The dataset contains 960 annotated images across 3 classes.
- Annotation tool: Roboflow
- Format: YOLOv5
- Train/Val/Test split: 70/20/10

Full details in `docs/DATASET.md`.

## Training
Training was done using Google Colab and the Roboflow YOLOv5 notebook.

Key settings:
- Model: YOLOv5s and YOLOv5x
- Epochs: 200
- Image size: 640
- Optimizer: SGD
- Dataset size: 960 images

See `notebooks/Train_YOLOv5.ipynb` for full training notebook.

## Results

### Model Accuracy
| Class         | YOLOv5s | YOLOv5x |
|---------------|---------|---------|
| Arduino Uno   | 95.2%   | 97.1%   |
| Motor Driver  | 93.9%   | 98.0%   |
| Servo Motor   | 95.7%   | 100%    |

More results (confusion matrix, loss curves) in `docs/RESULTS.md`.

![YOLOv5 Output](assets/results/detection1.jpg)

## Installation

git clone https://github.com/<your-username>/<repo>
cd <repo>

pip install -r requirements.txt

## Inference

### Image
python src/detect.py --weights models/yolov5s_best.pt --source assets/sample.jpg

### Webcam
python src/camera_detect.py --weights models/yolov5s_best.pt

## Project Structure

component-detection/
│── README.md
│── requirements.txt
│── src/
│── docs/
│── notebooks/
│── assets/
│── models/

## License
This project is licensed under the MIT License.



