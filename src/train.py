from ultralytics import YOLO

def train_model(data_yaml_path, epochs=100, img_size=640, model_name='yolov8n.pt'):
    """
    Trains a YOLOv8 model on a custom dataset.

    Args:
        data_yaml_path (str): Path to the dataset.yaml file.
                              The yaml file should look like this:
                              path: ../datasets/coco128  # dataset root dir
                              train: images/train2017  # train images (relative to 'path')
                              val: images/train2017  # val images (relative to 'path')
                              test:  # test images (optional)

                              names:
                                0: license_plate
        epochs (int): Number of training epochs.
        img_size (int): Image resolution.
        model_name (str): Base model to start training from (e.g., 'yolov8n.pt', 'yolov8s.pt').
    """
    # Load a model
    model = YOLO(model_name)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data=data_yaml_path, epochs=epochs, imgsz=img_size)

    print("Training complete.")
    print(f"Results saved in: {results.save_dir}")
    print(f"Best model weights: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    # Example usage:
    # train_model('path/to/data.yaml')
    print("This script is for training a custom YOLOv8 model.")
    print("Please provide a valid data.yaml file path to the train_model function.")
    print("Example: train_model('data.yaml', epochs=50)")
