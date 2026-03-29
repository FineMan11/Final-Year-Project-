# Component Detection for Product Assembly Using YOLOv5

> **Final Year Project** — Faculty of Electrical Engineering, Universiti Teknologi Malaysia (UTM)

| | |
|---|---|
| **Author** | Rais Hamizan Bin Faridan |
| **Supervisor** | PM Dr. Siti Armiza Mohd Aris |
| **University** | Universiti Teknologi Malaysia (UTM) |
| **Completed** | January 2025 |

---

## Overview

This project develops a real-time component detection system for product assembly in a manufacturing environment using YOLOv5 deep learning models. The system detects three electronic components — **Arduino Uno**, **Servo Motor**, and **Motor Driver** — from camera feeds on an assembly line.

The work was motivated by limitations in the existing template matching system used at Flex (manufacturing partner), which fails under rotation, scale changes, and lighting variation. YOLOv5 provides a robust, flexible, and fast alternative.

---

## Problem Background

The existing vision system at the manufacturing facility uses **template matching** — a classical computer vision approach that:
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

Three dataset sizes were evaluated to study the effect of training data volume:
- **Experiment 1:** 330 images (Arduino Uno + Servo Motor only)
- **Experiment 2:** 500 images (all 3 classes)
- **Experiment 3:** 960 images (all 3 classes, full dataset)

---

## Model Architecture

### YOLOv5s (Small)
- Lightweight, optimised for speed
- Suitable for edge deployment and real-time inference
- Lower parameter count, faster inference, lower resource usage

### YOLOv5x (Extra-Large)
- Highest accuracy in the YOLOv5 family
- Deeper backbone and more parameters
- Higher resource usage and slower inference speed

Both models were trained using **transfer learning** from COCO pre-trained weights.

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

### Latency Comparison (Roboflow Deployment Platform)

| Model | Latency | FPS |
|-------|---------|-----|
| YOLOv5s | 83.33 ms | ~12 fps |
| YOLOv5x | 333 ms | ~3 fps |

### Key Findings

- **YOLOv5x** achieves higher accuracy across all classes but is **4x slower** than YOLOv5s
- **YOLOv5s** at 960 images achieves 95%+ accuracy with ~12 fps, suitable for real-time assembly line use
- Adding more training data (330 to 960 images) consistently improves model performance
- The system successfully replaces the template matching approach with a more robust solution

---

## Project Structure



---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- pip

### Setup



---

## Running Inference

Use the Main Coding script to run detection on images or video:



### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| --weights | best.pt | Path to trained model weights |
| --source | data/images | Input source (image, video, or webcam index) |
| --conf-thres | 0.25 | Confidence threshold |
| --iou-thres | 0.45 | IoU threshold for NMS |
| --device | auto | CUDA device index or cpu |
| --view-img | False | Display results in window |
| --save-txt | False | Save results to .txt files |
| --save-conf | False | Save confidence scores in labels |

---

## Training Your Own Model

1. Collect images of your components and annotate them using Roboflow (https://roboflow.com)
2. Export the dataset in YOLOv5 PyTorch format
3. Upload to Google Colab and run:



---

## References

1. Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767
2. Jocher, G. et al. (2020). ultralytics/yolov5. GitHub. https://github.com/ultralytics/yolov5
3. Lin, T. Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014
4. Roboflow. (2021). Roboflow: Give your software the sense of sight. https://roboflow.com
5. Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). YOLOv4. arXiv:2004.10934
6. Ren, S., et al. (2015). Faster R-CNN. NeurIPS 2015
7. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR 2016
8. Goodfellow, I., et al. (2016). Deep Learning. MIT Press
9. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444
10. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks. arXiv:1409.1556

---

## License

This project is for academic purposes. All rights reserved by the author.

---

## Contact

**Rais Hamizan Bin Faridan**
Faculty of Electrical Engineering
Universiti Teknologi Malaysia (UTM)
GitHub: https://github.com/FineMan11
