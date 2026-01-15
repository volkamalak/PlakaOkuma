import cv2
from ultralytics import YOLO
import os

class PlateDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the PlateDetector with a YOLOv8 model.

        Args:
            model_path (str): Path to the .pt model file.
                              Default is 'yolov8n.pt' (standard COCO model).
                              For best results, train a model on license plates
                              and use the path to 'best.pt'.
        """
        if not os.path.exists(model_path) and not model_path.endswith('.pt'):
             # Ultralytics will download standard models automatically,
             # but custom paths must exist.
             raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(model_path)
        # Class IDs for COCO dataset: 2=car, 3=motorcycle, 5=bus, 7=truck
        self.vehicle_classes = [2, 3, 5, 7]

    def detect(self, image_path_or_array, conf_threshold=0.25):
        """
        Detects objects (ideally plates, or vehicles if using standard model) in the image.

        Args:
            image_path_or_array: File path or numpy array of the image.
            conf_threshold (float): Confidence threshold for detection.

        Returns:
            list: A list of detection dictionaries containing:
                  {'box': [x1, y1, x2, y2], 'conf': float, 'class': int, 'name': str}
        """
        results = self.model(image_path_or_array, conf=conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                name = result.names[cls]

                # If using a custom model trained only on plates,
                # we return everything.
                # If using standard YOLO, we might want to filter for vehicles
                # BUT the user wants Plate Detection.
                # Ideally, this model IS the plate detector.
                # For now, we append everything and let the main app decide or
                # assume the model is correct.

                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'conf': conf,
                    'class': cls,
                    'name': name
                })

        return detections
