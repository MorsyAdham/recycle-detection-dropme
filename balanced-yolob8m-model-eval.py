import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import seaborn as sns
import torch
from pathlib import Path

# --- Configuration ---
model_path = 'yolov8m-balanced-dataset.pt'
data_yaml = 'data/yolo-extracted/data.yaml'
save_dir = 'balanced_evaluation_results'
confidence_threshold = 0.5

os.makedirs(save_dir, exist_ok=True)

# --- Load YOLO model ---
model = YOLO(model_path)

# --- Run evaluation ---
metrics = model.val(data=data_yaml, save=True, save_json=True, save_dir=save_dir)
