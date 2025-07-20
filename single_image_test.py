import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# 1. Paths and setup
model_path = 'yolov8m-balanced-dataset.pt'  # Path to your trained model
image_path = 'data/yolo-extracted/images/test/1a2298e9-image_2025-03-12_13-37-45_aug0.jpg'  # Path to the image
save_dir = 'evaluation_results'               # Directory to save results
confidence_threshold = 0.5                    # Confidence threshold for 7-class detection

os.makedirs(save_dir, exist_ok=True)

# 2. Load the model
model = YOLO(model_path)

# 3. Run inference
results = model(image_path)
result = results[0]  # First result (single image)

# 4. Print prediction details
print("/nValid Predictions (confidence ≥ {:.2f}):".format(confidence_threshold))
valid_boxes = []

for i, box in enumerate(result.boxes):
    conf = float(box.conf[0])
    if conf < confidence_threshold:
        continue  # Skip low-confidence predictions

    cls_id = int(box.cls[0])
    class_name = result.names[cls_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    print(f"{len(valid_boxes)+1}. Class: {class_name}, Confidence: {conf:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})")
    valid_boxes.append((x1, y1, x2, y2, class_name, conf))

# 5. Draw only valid predictions on the image
image = cv2.imread(image_path)
for x1, y1, x2, y2, class_name, conf in valid_boxes:
    label = f"{class_name}: {conf:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 0, 0), 2)

# Save the filtered result
result_img_path = os.path.join(save_dir, 'predicted_filtered.jpg')
cv2.imwrite(result_img_path, image)

# 6. Show original and filtered predicted images side by side
original = cv2.imread(image_path)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
filtered = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered)
plt.title('Filtered Predictions (≥ {:.2f})'.format(confidence_threshold))
plt.axis('off')

plt.tight_layout()
plt.show()

# 7. Warning if no predictions passed the threshold
if not valid_boxes:
    print("⚠️ No predictions passed the confidence threshold.")
