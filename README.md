# Component Detection for Product Assembly in Manufacturing Using YOLOv5

This repository contains the implementation of a deep learning–based component detection system for manufacturing environments.  
The model is built using **YOLOv5** and is designed to detect three main components:

- Arduino Board  
- Printed Circuit Board (PCB)  
- Arduino Nano  

This project is part of my **Final Year Project (FYP)** at **Universiti Teknologi Malaysia (UTM)**.

---

## 📌 Objectives

- Develop a YOLOv5 object detection model optimized for manufacturing components.
- Achieve real-time detection (within 1 second).
- Improve the accuracy and reliability of automated assembly processes.
- Provide an easily deployable model for companies such as **Flex**.

---

## 📂 Project Structure


---

## 🖼️ Dataset & Annotation Format

Images were collected and manually annotated using **MakeSense.ai**.

YOLO annotation format:


Where:

- `class` → numerical label (0 = Arduino board, 1 = PCB, 2 = Arduino Nano)  
- `x_center` → normalized center x-coordinate (0–1)  
- `y_center` → normalized center y-coordinate (0–1)  
- `width` → bounding box width (normalized)  
- `height` → bounding box height (normalized)

### Example annotation:

---

## 🧰 Tools Used

- **YOLOv5 by Ultralytics**
- **Google Colab** (GPU training)
- **MakeSense.ai** (annotation tool)
- **Python / PyTorch**
- **OpenCV**

---

## 🚀 Model Training

### Clone YOLOv5
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
| Metric    | Result                                             |
| --------- | -------------------------------------------------- |
| Precision | High                                               |
| Recall    | High                                               |
| mAP@0.5   | ~0.995                                             |
| F1 Score  | High                                               |
| Issue     | Some false positives (Arduino board vs background) |
python detect.py --weights best.pt --img 640 --source your_image.jpg
