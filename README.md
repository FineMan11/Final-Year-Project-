# Component Detection for Product Assembly in Manufacturing Using Deep Learning (YOLOv5)

This project focuses on developing a deep learning–based object detection model using **YOLOv5** to identify manufacturing components such as:

- Arduino Board  
- Arduino Nano  
- Printed Circuit Board (PCB)

The project was developed as part of the Final Year Project (FYP) for the Bachelor of Electrical Engineering (Mechatronics) at **Universiti Teknologi Malaysia (UTM)**.

---

## 📌 Project Objective

To design and develop a fast and accurate object detection model capable of identifying manufacturing components **within 1 second**, to support automation in industries such as Flex.

---

## 📁 Project Structure

---

## 🔧 Tools & Technologies

- **YOLOv5 (Ultralytics)**
- **Python**
- **PyTorch**
- **Google Colab**
- **Makesense.ai** (image annotation)
- **Kaggle Dataset** (50+ images)

---

## 🧠 Methodology

### 1. **Data Collection**
- 50+ component images collected from Kaggle  
- Components include Arduino boards and PCBs

### 2. **Data Preprocessing**
- Images resized to **640×640**
- Normalization applied
- Data augmentation (rotation, flip, brightness)
- Annotation using Makesense.ai  
- YOLO format:  

