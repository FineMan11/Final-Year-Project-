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
- Annotation format:

Where:

- `class` → numerical label (0 = Arduino board)
- `x_center` → center x-coordinate (normalized 0–1)
- `y_center` → center y-coordinate (normalized 0–1)
- `width` → box width (normalized)
- `height` → box height (normalized)

**Example:class x_center y_center width height**