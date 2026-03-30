# Component Detection for Product Assembly Using YOLOv5

> **Final Year Project** - Faculty of Electrical Engineering, Universiti Teknologi Malaysia (UTM)

| | |
|---|---|
| **Author** | Rais Hamizan Bin Faridan |
| **Supervisor** | Dr. Nurulaqilla Binti Khamis |
| **University** | Universiti Teknologi Malaysia (UTM) |
| **Completed** | January 2025 |

---

## Overview

This project develops a real-time component detection system for product assembly in a manufacturing environment using YOLOv5 deep learning models. The system detects three electronic components - **Arduino Uno**, **Servo Motor**, and **Motor Driver** - from camera feeds on an assembly line.

The work was motivated by limitations in the existing template matching system used at Flex (manufacturing partner), which fails under rotation, scale changes, and lighting variation. YOLOv5 provides a robust, flexible, and fast alternative.

---

## Sample Detection Results

![Detection Sample - Arduino Uno](images/detection_sample_arduino.jpg)
*YOLOv5 detecting Arduino Uno with bounding box and confidence score*

![Dataset Sample - All Classes](images/dataset_sample.jpg)
*Annotated dataset sample showing all 3 component classes*

---

## Problem Background

The existing vision system at the manufacturing facility uses **template matching** which:
- Fails when components are rotated or scaled differently from the template
- Is sensitive to changes in lighting conditions
- Requires manual re-configuration when new product variants are introduced
- Cannot handle partial occlusion or component overlap

A deep learning-based detection approach using YOLOv5 overcomes all of these limitations.

---

## Objectives

1. Develop a YOLOv5-based object detection model to detect electronic components in product assembly
2. Compare the performance of **YOLOv5s** (small/fast) and **YOLOv5x** (extra-large/accurate) models
3. Evaluate detection accuracy across varying dataset sizes (330, 500, and 960 images)

---

## Dataset

| Property | Details |
|----------|---------|
| **Classes** | Arduino Uno, Servo Motor, Motor Driver |
| **Total Images** | ~960 images |
| **Collection Method** | Video frame extraction at 60 fps |
| **Annotation Tool** | Roboflow (bounding box labelling) |
| **Data Split** | 80% training / 20% validation |
| **Export Format** | YOLOv5 PyTorch format via Roboflow |

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| **Framework** | Ultralytics YOLOv5 |
| **Platform** | Google Colab (T4 GPU) |
| **Batch Size** | 16 |
| **Epochs** | 200 |
| **Image Size** | 640x640 |
| **Optimiser** | SGD |
| **Pre-trained Weights** | YOLOv5s.pt / YOLOv5x.pt (COCO) |
| **Dataset Management** | Roboflow |

---

## Results

### Accuracy by Dataset Size (YOLOv5s)

| Class | 330 Images | 500 Images | 960 Images |
|-------|-----------|-----------|-----------|
| Arduino Uno | 96.8% | 100% | 95.2% |
| Motor Driver | - | 95.8% | 93.9% |
| Servo Motor | 96.4% | 97.2% | 95.7% |

> Motor Driver was not included in Experiment 1 (330 images).

### YOLOv5s vs YOLOv5x (960 Images)

| Class | YOLOv5s | YOLOv5x |
|-------|---------|---------|
| Arduino Uno | 95.2% | 97.1% |
| Motor Driver | 93.9% | 98.0% |
| Servo Motor | 95.7% | 100% |

### Latency Comparison

| Model | Latency | FPS |
|-------|---------|-----|
| YOLOv5s | 83.33 ms | ~12 fps |
| YOLOv5x | 333 ms | ~3 fps |

### Training Loss Curves

![Training Loss YOLOv5s](images/training_loss_yolov5s.png)
*Loss curves over 200 epochs - YOLOv5s (960 images)*

![Training Loss YOLOv5x](images/training_loss_yolov5x.png)
*Loss curves over 200 epochs - YOLOv5x (960 images)*

### mAP Curves

![mAP YOLOv5s](images/map_yolov5s.png)
*mAP@0.5 over training - YOLOv5s*

![mAP YOLOv5x](images/map_yolov5x.png)
*mAP@0.5 over training - YOLOv5x*

### Confusion Matrices

![Confusion Matrix Exp1](images/confusion_matrix_exp1.png)
*Experiment 1 - YOLOv5s, 330 images*

![Confusion Matrix Exp2](images/confusion_matrix_exp2.png)
*Experiment 2 - YOLOv5s, 500 images*

![Confusion Matrix Exp3](images/confusion_matrix_exp3.png)
*Experiment 3 - YOLOv5s, 960 images*

![Confusion Matrix Exp4](images/confusion_matrix_exp4.png)
*Experiment 4 - YOLOv5x, 960 images*

### Key Findings

- **YOLOv5x** achieves higher accuracy across all classes but is **4x slower** than YOLOv5s
- **YOLOv5s** at 960 images achieves 95%+ accuracy with ~12 fps, suitable for real-time assembly line use
- Adding more training data consistently improves model performance
- The system successfully replaces template matching with a more robust solution

---

## Project Structure

```
Final-Year-Project-/
├── README.md           # Project overview and results
├── Testing.md          # Testing methodology and experiment details
├── results             # Detailed results summary
├── Main Coding         # Python inference script
├── requirements.txt    # Python dependencies
└── images/             # Figures, detection samples, training plots
    └── README.md       # Guide to image assets
```

---

## Installation

```bash
git clone https://github.com/FineMan11/Final-Year-Project-.git
cd Final-Year-Project-
pip install -r requirements.txt
```

---

## Running Inference

```bash
# Single image
python "Main Coding" --weights best.pt --source image.jpg

# Video file
python "Main Coding" --weights best.pt --source video.mp4

# Webcam
python "Main Coding" --weights best.pt --source 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| --weights | best.pt | Path to trained model weights |
| --source | data/images | Image, video, folder, or webcam index |
| --conf-thres | 0.25 | Confidence threshold |
| --iou-thres | 0.45 | IoU NMS threshold |
| --device | auto | CUDA device index or cpu |
| --view-img | False | Display results in window |
| --save-txt | False | Save results to .txt files |

---

## License

This project is for academic purposes. All rights reserved by the author.

---

## Contact

**Rais Hamizan Bin Faridan**
Faculty of Electrical Engineering, Universiti Teknologi Malaysia (UTM)
GitHub: https://github.com/FineMan11
