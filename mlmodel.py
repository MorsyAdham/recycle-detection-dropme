from enum import IntEnum
from typing import NamedTuple

import cv2
from ultralytics import YOLO


class Item(IntEnum):
    ALUMINUM = 0
    PLASTIC = 1
    BACKGROUND = 2
    HAND = 3


class Decision(IntEnum):
    ACCEPTED = 0
    REJECTED = 1


class Result(NamedTuple):
    bounding_box: tuple[int, int, int, int]
    confidence: float
    item: Item
    decision: Decision


class MLModel:
    def __init__(self, yolo_model_path: str) -> None:
        self.model = YOLO(yolo_model_path)

    def predict(self, image_path: str, prediction_image: str | None = None) -> list[Result]:
        results = self.model(image_path)
        result = results[0]  # First result (single image)

        valid_boxes = []
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            if conf < 0.05:
                continue  # Skip low-confidence predictions

            try:
                cls_id = int(box.cls[0])
            except IndexError:
                return []
            class_name = result.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            valid_boxes.append((x1, y1, x2, y2, class_name, conf))

        # 5. Draw only valid predictions on the image
        image = cv2.imread(image_path)
        for x1, y1, x2, y2, class_name, conf in valid_boxes:
            label = f"{class_name}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 0, 0), 2)

        # Save the filtered result
        cv2.imwrite(prediction_image, image)

        ret = []
        for res in results:
            box = res.boxes
            
            try:
                cls_id = int(box.cls[0]) # e.g., 0, 1, 2, ...
            except IndexError:
                return []         
            conf = float(box.conf[0])        # e.g., 0.86
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordinates
            decision = Decision.REJECTED
            if cls_id == 0 or cls_id == 1:
                decision = Decision.ACCEPTED
            if cls_id >= 4:
                item = None
            else:
                item = Item(cls_id)
            ret.append(Result(
                (x1, y1, x2, y2),
                conf,
                item,
                decision,
            ))
        return ret
