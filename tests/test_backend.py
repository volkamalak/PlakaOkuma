import unittest
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector import PlateDetector

class TestPlateDetector(unittest.TestCase):
    def setUp(self):
        pass

    def test_init_defaults(self):
        """Test initialization with default behavior."""
        try:
            detector = PlateDetector()
            self.assertIsNotNone(detector.model)
        except Exception as e:
            self.fail(f"Initialization failed: {e}")

    def test_detect_dummy_image(self):
        """Test detection on a black image."""
        try:
            detector = PlateDetector()
        except Exception as e:
            print(f"Skipping detection test due to model load failure: {e}")
            return

        # Create a black image (640x640)
        img = np.zeros((640, 640, 3), dtype=np.uint8)

        results = detector.detect(img)

        # We expect a list (empty or not)
        self.assertIsInstance(results, list)

        # On a black image, result should likely be empty unless phantom detection
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
