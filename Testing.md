# Testing Documentation

## Overview

This document describes the testing methodology, experimental setup, evaluation metrics, and results for the Component Detection for Product Assembly project using YOLOv5.

---

## Experimental Setup

Four experiments were conducted to evaluate model performance:

| Experiment | Model | Dataset Size | Classes |
|-----------|-------|-------------|---------|
| 1 | YOLOv5s | 330 images | Arduino Uno, Servo Motor |
| 2 | YOLOv5s | 500 images | Arduino Uno, Servo Motor, Motor Driver |
| 3 | YOLOv5s | 960 images | Arduino Uno, Servo Motor, Motor Driver |
| 4 | YOLOv5x | 960 images | Arduino Uno, Servo Motor, Motor Driver |

All training was performed on **Google Colab** using a **T4 GPU** with the following fixed hyperparameters:
- Batch size: 16
- Epochs: 200
- Image size: 640x640
- Optimiser: SGD
- Pre-trained weights: COCO (transfer learning)

---

## Dataset Preparation

### Data Collection
- Components were filmed individually on an assembly board
- Frames were extracted from video at 60 fps
- Resulting in approximately 960 usable images

### Annotation
- All images annotated using **Roboflow** with bounding boxes
- Labels: , , 
- Exported in YOLOv5 PyTorch format

### Data Split
- Training: 80%
- Validation: 20%

---

## Evaluation Metrics

The following metrics were used to evaluate model performance:

### Detection Accuracy Metrics
| Metric | Description |
|--------|-------------|
| **Precision** | Ratio of true positive detections to all positive detections |
| **Recall** | Ratio of true positive detections to all actual objects |
| **mAP@0.5** | Mean Average Precision at IoU threshold of 0.50 |
| **mAP@0.5:0.95** | Mean Average Precision averaged over IoU thresholds 0.50 to 0.95 |

### Training Loss Metrics
| Loss | Description |
|------|-------------|
| **Box Loss** | Measures bounding box localisation accuracy |
| **Objectness Loss** | Measures confidence in object presence |
| **Classification Loss** | Measures class prediction accuracy |

### Latency
- Measured on **Roboflow deployment platform**
- Reported as milliseconds per frame and frames per second (fps)

---

## Experiment 1: YOLOv5s — 330 Images

### Dataset
- Arduino Uno: 165 images
- Servo Motor: 165 images
- Total: 330 images (2 classes only, Motor Driver not included)

### Results

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Arduino Uno | 0.968 | 0.962 | 0.968 |
| Servo Motor | 0.964 | 0.959 | 0.964 |
| **Overall** | **0.966** | **0.961** | **0.966** |

### Training Observations
- Model converged steadily over 200 epochs
- Box loss and classification loss decreased consistently
- Both classes achieved high detection accuracy (>96%)

---

## Experiment 2: YOLOv5s — 500 Images

### Dataset
- Arduino Uno: ~167 images
- Servo Motor: ~167 images
- Motor Driver: ~166 images
- Total: 500 images (3 classes)

### Results

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Arduino Uno | 1.000 | 0.997 | 1.000 |
| Motor Driver | 0.958 | 0.951 | 0.958 |
| Servo Motor | 0.972 | 0.968 | 0.972 |
| **Overall** | **0.977** | **0.972** | **0.977** |

### Training Observations
- Adding Motor Driver class increased dataset diversity
- Arduino Uno achieved perfect 100% precision in this experiment
- All three classes performed at or above 95.8% accuracy

---

## Experiment 3: YOLOv5s — 960 Images

### Dataset
- Arduino Uno: ~320 images
- Servo Motor: ~320 images
- Motor Driver: ~320 images
- Total: 960 images (3 classes, full dataset)

### Results

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Arduino Uno | 0.952 | 0.948 | 0.952 |
| Motor Driver | 0.939 | 0.934 | 0.939 |
| Servo Motor | 0.957 | 0.952 | 0.957 |
| **Overall** | **0.949** | **0.945** | **0.949** |

### Training Observations
- Larger dataset introduced more variation, causing slight accuracy drop vs. 500 images
- However, model is more generalisable with a larger training set
- All classes maintained accuracy above 93.9%

---

## Experiment 4: YOLOv5x — 960 Images

### Dataset
- Same 960-image dataset as Experiment 3
- Model: YOLOv5x (extra-large, highest accuracy variant)

### Results

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Arduino Uno | 0.971 | 0.967 | 0.971 |
| Motor Driver | 0.980 | 0.977 | 0.980 |
| Servo Motor | 1.000 | 0.998 | 1.000 |
| **Overall** | **0.984** | **0.981** | **0.984** |

### Training Observations
- YOLOv5x achieved the best accuracy across all 4 experiments
- Servo Motor reached perfect 100% detection
- Motor Driver and Arduino Uno both exceeded 97% accuracy
- Training took significantly longer due to larger model size

---

## Model Comparison: YOLOv5s vs YOLOv5x

### Accuracy Comparison (960 Images)

| Class | YOLOv5s | YOLOv5x | Improvement |
|-------|---------|---------|-------------|
| Arduino Uno | 95.2% | 97.1% | +1.9% |
| Motor Driver | 93.9% | 98.0% | +4.1% |
| Servo Motor | 95.7% | 100% | +4.3% |
| **Average** | **94.9%** | **98.4%** | **+3.5%** |

### Latency Comparison

| Model | Latency per Frame | FPS | Relative Speed |
|-------|------------------|-----|----------------|
| YOLOv5s | 83.33 ms | ~12 fps | 4x faster |
| YOLOv5x | 333 ms | ~3 fps | Baseline |

---

## Confusion Matrix Analysis

### YOLOv5s (960 Images)
- Arduino Uno: Correctly classified in ~95% of cases
- Motor Driver: Correctly classified in ~94% of cases, some false negatives at occlusion
- Servo Motor: Correctly classified in ~96% of cases

### YOLOv5x (960 Images)
- Arduino Uno: Correctly classified in ~97% of cases
- Motor Driver: Correctly classified in ~98% of cases
- Servo Motor: Correctly classified in ~100% of cases

---

## Key Conclusions

1. **More data generally improves performance** — training with 960 images produced a more robust model than 330 or 500 images, despite a slight accuracy dip on validation metrics

2. **YOLOv5x is more accurate but significantly slower** — 98.4% average accuracy vs. 94.9% for YOLOv5s, but 4x the inference latency (333ms vs. 83ms)

3. **YOLOv5s is suitable for real-time deployment** — at ~12 fps, it can support real-time assembly line monitoring with acceptable accuracy

4. **The system outperforms template matching** — template matching fails under rotation and lighting changes, while YOLOv5 handles these conditions robustly

5. **Transfer learning is effective** — starting from COCO pre-trained weights allowed fast convergence with a relatively small domain-specific dataset
