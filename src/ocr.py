import cv2
import easyocr
import numpy as np

class PlateReader:
    def __init__(self, languages=['en']):
        """
        Initializes the EasyOCR reader.

        Args:
            languages (list): List of language codes. Default is ['en'].
                              EasyOCR supports many languages.
        """
        # gpu=False for compatibility in sandbox environment,
        # but set to True if CUDA is available in production.
        self.reader = easyocr.Reader(languages, gpu=False)

    def preprocess(self, image):
        """
        Applies preprocessing to the image to improve OCR accuracy.

        Args:
            image (numpy array): Input image (BGR).

        Returns:
            numpy array: Preprocessed image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding or other filters if necessary
        # Otsu's thresholding
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Sometimes raw grayscale is better for EasyOCR than binary thresholding
        # depending on lighting. Let's return grayscale for now, or
        # could try adaptive thresholding.

        return gray

    def read_text(self, image_crop):
        """
        Reads text from a cropped license plate image.

        Args:
            image_crop (numpy array): Cropped image of the license plate.

        Returns:
            list: List of tuples (bbox, text, prob).
        """
        if image_crop is None or image_crop.size == 0:
            return []

        processed_img = self.preprocess(image_crop)

        # detail=1 returns bounding box, text, and confidence
        results = self.reader.readtext(processed_img, detail=1)

        return results
