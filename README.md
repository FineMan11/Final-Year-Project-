# Component Detection for Product Assembly Using YOLOv5  
Automated detection of Arduino Uno, Motor Driver, and Servo Motor to support assembly verification in manufacturing environments.

---

## 🚀 Demo  
*(Replace with your own demo GIF or image)*

![Demo](assets/demo.gif)

---

## 📑 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Installation](#installation)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [License](#license)
- [Contact](#contact)

---

## 🔍 Overview  
This project implements a deep-learning–based system to automatically detect electronic components commonly used in product assembly:

- **Arduino Uno**  
- **Motor Driver**  
- **Servo Motor**

It aims to enhance **manufacturing quality control** by enabling machines to validate components during the assembly process, reducing human error and improving production efficiency.

This is my Final Year Project at **Universiti Teknologi Malaysia (UTM)** titled:

> **Component Detection for Product Assembly in Manufacturing Using Deep Learning Model**

The model was trained using **YOLOv5**, chosen for its balance of speed, accuracy, and suitability for real-time applications.

---

## ⭐ Features  
- 🧠 Trained **YOLOv5s** and **YOLOv5x** models  
- 📊 Dataset of **960 annotated images** (Roboflow)  
- 🎥 Real-time inference support (webcam)  
- 🔍 High accuracy: Servo Motor detection reached **100%**  
- ⚡ Lightweight model available for fast on-device detection  
- 🌐 Flask API included for deployment  
- 📝 Extensive documentation (dataset, experiments, results)

---

## 📁 Dataset  
A total of **960 images** were collected and annotated into 3 classes:

| Class | Description |
|-------|-------------|
| Arduino Uno | Main microcontroller board |
| Motor Driver | Motor driver module |
| Servo Motor | Standard servo motor |

**Dataset details:**
- Annotation Tool: **Roboflow**  
- Format: **YOLOv5** (train/val/test folders)  
- Split: **70% train**, **20% validation**, **10% test**  
- Preprocessing: Auto-orient, resize, augmentation  

📄 Full dataset documentation → `docs/DATASET.md`

---

## 🏋️‍♂️ Training  
Training was performed in **Google Colab** using Roboflow’s YOLOv5 notebook.

**Training settings:**

| Parameter | Value |
|----------|--------|
| Model | YOLOv5s & YOLOv5x |
| Epochs | 200 |
| Image size | 640×640 |
| Optimizer | SGD |
| Batch size | 16 |
| Dataset size | 960 |

Training notebook → `notebooks/Train_YOLOv5.ipynb`  
Training setup → `docs/TRAINING_SETUP.md`

---

## 📊 Results  

### **Model Accuracy**
| Class | YOLOv5s | YOLOv5x |
|--------|---------|---------|
| Arduino Uno | 95.2% | 97.1% |
| Motor Driver | 93.9% | 98.0% |
| Servo Motor | 95.7% | 100% |

### **Latency (Roboflow Benchmarks)**  
- **YOLOv5s:** ~83 ms (≈ 12 FPS) → Fastest  
- **YOLOv5x:** ~333 ms (≈ 3 FPS) → Most accurate  

📸 Detection example:

*(Placeholders: replace with your images)*

