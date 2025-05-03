
# ♻️ Drop Me - Recyclable Material Detection

This project aims to train an object detection model capable of identifying recyclable materials inside a smart **Recycle Vending Machine** developed by **Drop Me**. The model is trained using a **custom YOLOv8 object detection pipeline** built on top of **Lightning AI** – a powerful platform that enables efficient, scalable, and reproducible AI development.

* The entire training pipeline, preprocessing, and model experiments are managed through [**Lightning AI (lightning.ai)**](https://lightning.ai/), which ensures:

  * Seamless hardware acceleration (multi-GPU, TPU, etc.)
  * Reproducibility and experiment versioning
  * Cleaner model and data separation for modularity and scalability

The data used is custom-collected from inside the vending machine and categorized into six key classes.

---

## 📁 Project Structure

```
DropMe-Recycle-Detection/
│
├── data/
│   ├── yolo-extracted/          # Dataset unbalanced, processed dataset
│   ├── yolo-extracted-balanced/ # Final balanced, processed dataset
│   └── yolo-new-dataset.zip     # Data compressed 
├── runs/                        # YOLO training results and models
├── data_handling.ipynb          # Data processing and preprocessing
├── yolo8_model_training.ipynb   # YOLOv8 model configuration and training
└── README.md                    # Project overview and instructions
```

---

## 📦 Dataset Information

### 🔢 Classes (6 total)

The project includes the following six classes:

1. `accepted_aluminium`
2. `accepted_plastic`
3. `background`
4. `hand`
5. `rejected_aluminium`
6. `rejected_plastic`

### 🧾 YOLO Format

Each `.txt` label file contains lines in the format:

```
class_id center_x center_y width height
```

* All values are **normalized (0–1)** relative to the image size.
* Labels match images by filename (e.g., `img001.png` → `img001.txt`).

---

## 📊 Data Handling & Preprocessing (data\_handling.ipynb)

The preprocessing pipeline ensures clean, high-quality training data for the model. It includes:

### 1. 🔓 Unzipping

Initial dataset is extracted from the original archive.

### 2. 🧪 Train/Test/Validation Split

The dataset is split into:

* **70% Training**
* **20% Validation**
* **10% Testing**

Each phase gets separate subfolders for `images/` and `labels/`.

### 3. 🧼 Cleaning

* Verifies label/image correspondence.
* Removes mislabeled or empty files.
* Ensures all YOLO labels are correctly formatted.

### 4. 🎨 Data Augmentation

Used to improve model generalization. Includes:

* **Rotation**
* **Flipping**
* **Color Jitter**
* **Random Crop**

✅ **Note**: All label coordinates are correctly **transformed** alongside image augmentations (e.g., after rotation), preserving YOLO format integrity.

### 5. ⚖️ Class Balancing

Some classes (e.g., `background`, `hand`) may dominate the dataset.

To resolve this:

* **Undersampling**: Limits number of samples for overrepresented classes.
* **Oversampling**: Augments underrepresented classes to match distribution.

Final dataset is saved to:

```
yolo-extracted-balanced/
```

---

## ⚙️ YOLOv8 Model Training (yolo8\_model\_training.ipynb)

### 🧠 Model Setup

* Model used: **YOLOv8 (Ultralytics)**
* Framework: `ultralytics` Python package
* Data: `data/yolo-extracted-balanced/data.yaml`

### 🔧 Training Configuration

```python
results = model.train(
    data='data/yolo-extracted-balanced/data.yaml',
    epochs=200,
    imgsz=768,
    batch=16,
    cache=True,
    workers=8,
    amp=True,
    device=device,  # e.g. 'cuda' or 'cpu'
    lr0=0.005,
    lrf=0.0005,
    warmup_epochs=5,
    warmup_momentum=0.75,
    optimizer='AdamW',
    weight_decay=0.0005,
    val=True,
    save_period=5,
    patience=15,
    cos_lr=True
)
```

### 💾 Custom Save Name for the Model

To change the model name (e.g., `dropme-recycle-v1.pt`), you can **manually rename** the saved model inside the `runs/detect/train/weights` directory **after training**.

Alternatively, move it programmatically:

```python
import shutil
shutil.copy('runs/detect/train/weights/best.pt', 'models/dropme-recycle-v1.pt')
```

---

## 📈 Outputs

* Model weights: `runs/detect/train/weights/best.pt`
* Training logs, metrics, loss curves: `runs/detect/train/`
* Validation predictions and visualizations: Automatically saved every few epochs.

---

## 🚀 How to Run the Project

### 🐍 Requirements

```bash
pip install ultralytics opencv-python matplotlib
```

### 🔧 Step-by-Step

1. **Prepare the dataset**
   Open and run `data_handling.ipynb`
   This will preprocess, augment, balance, and save to `yolo-extracted-balanced/`.

2. **Train the model**
   Open and run `yolo8_model_training.ipynb`.
   Adjust parameters if needed. The best model will be saved.

3. **Use or Evaluate the model**
   You can test the model using:

   ```python
   model = YOLO('runs/detect/train/weights/best.pt')
   model.predict('path/to/image.png', save=True)
   ```

---

## 🧠 Project Purpose & Context

This object detection model is part of a broader solution by **Drop Me**, a company focused on developing smart environmental solutions. This specific use-case involves detecting recyclables placed inside an **automated recycling vending machine**.

The model allows the system to:

* Identify whether the inserted material is recyclable.
* Classify it as accepted or rejected material.
* Detect hands (for safety mechanisms).
* Handle edge cases using background detection.

---

## 📌 Final Notes

* Ensure your YOLO data format is preserved.
* For deployment, consider converting to **ONNX** or **TensorRT** for faster inference.
* Always test the model on **real-life machine camera feeds** for robustness.

---

Let me know if you want this exported to a `.md` file or used as the base for a formal PDF or web-based documentation.
