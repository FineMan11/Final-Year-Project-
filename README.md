# Component Detection for Product Assembly in Manufacturing Using Deep Learning (YOLOv5)

This project focuses on developing a deep learning–based object detection model using **YOLOv5** to identify components used in manufacturing assembly lines.  
The model was trained to detect three components, with the FYP1 preliminary results focusing on the **Arduino board**.


---

## 🧠 Project Goal

To develop a fast and accurate object detection model capable of identifying components **within 1 second**, improving manufacturing efficiency and reducing errors.

---

## 📦 Components Detected

- Arduino Board  
- PCB Board  
- Arduino Nano  

*(For FYP1, only Arduino board detection was trained and evaluated.)*

---

## 📂 Dataset

- **50 images** used for initial training  
- Collected from **Kaggle**  
- Images manually labeled using **MakeSense.ai**  
- Annotation format: class x_center y_center width height

Where:

- `class` → numerical label (0 = Arduino board)
- `x_center` → center x-coordinate (normalized 0–1)
- `y_center` → center y-coordinate (normalized 0–1)
- `width` → box width (normalized)
- `height` → box height (normalized)

**Example:0 0.509681 0.508036 0.820308 0.869643**

---

## 🧪 Model Training

The model was trained in Google Colab using the following hyperparameters:

- **Learning rate:** 0.001  
- **Batch size:** 16  
- **Epochs:** 50  
- **Model:** YOLOv5s (pretrained weights)

Training steps:

1. Clone YOLOv5 repo  
2. Prepare dataset (train/val split = 80/20)  
3. Train the model  
4. Evaluate performance using the built-in YOLOv5 metrics  

---

## 📊 Performance Results

The YOLOv5 model achieved:

- ✔️ **High precision**
- ✔️ **High recall**
- ✔️ **mAP@0.5 ≈ 0.995**
- ✔️ Good performance in early-stage testing

However, challenges include:

- ❗ False positives when detecting Arduino boards (background confusion)

Visual results included:

- Precision–Confidence Curve  
- Precision–Recall Curve  
- F1-Confidence Curve  
- Loss Curves  
- Confusion Matrix  

---

## 🚀 How to Run the Model

### **1️⃣ Clone YOLOv5**
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

/datasets/arduino/
    ├── images/
    │     ├── train
    │     └── val
    └── labels/
          ├── train
          └── val
