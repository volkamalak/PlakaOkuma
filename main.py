import cv2
import argparse
import os
from src.detector import PlateDetector
from src.ocr import PlateReader

def main():
    parser = argparse.ArgumentParser(description="License Plate Recognition System")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--output', type=str, default='data/output', help='Directory to save results')
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Load modules
    print(f"Loading detector with model: {args.model}...")
    try:
        detector = PlateDetector(model_path=args.model)
    except Exception as e:
        print(f"Error loading detector: {e}")
        return

    print("Loading OCR reader...")
    reader = PlateReader(languages=['en']) # Add more languages if needed

    # Load Image
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    img = cv2.imread(args.image)
    if img is None:
        print("Error: Could not read image.")
        return

    # 1. Detect
    print("Detecting plates...")
    detections = detector.detect(img)
    print(f"Found {len(detections)} objects.")

    # 2. Process and Read
    for i, det in enumerate(detections):
        box = det['box']
        conf = det['conf']
        cls_name = det['name']

        # Crop the detected region
        x1, y1, x2, y2 = box
        # Clip to image bounds
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        plate_crop = img[y1:y2, x1:x2]

        # 3. OCR
        # We only try to read if it's likely a plate.
        # If using standard YOLO, 'truck' or 'car' is not a plate.
        # But for this demo, we will attempt OCR on ALL detections
        # because the user might provide a cropped plate image OR
        # the model might be a custom plate model.

        print(f"Object {i}: {cls_name} ({conf:.2f})")
        ocr_results = reader.read_text(plate_crop)

        detected_text = ""
        if ocr_results:
            # Combine text parts if multiple
            detected_text = " ".join([res[1] for res in ocr_results])
            print(f"  -> Text: {detected_text}")
        else:
            print("  -> No text found.")

        # Draw on image
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {detected_text}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save result
    filename = os.path.basename(args.image)
    save_path = os.path.join(args.output, f"result_{filename}")
    cv2.imwrite(save_path, img)
    print(f"Result saved to {save_path}")

if __name__ == "__main__":
    main()
