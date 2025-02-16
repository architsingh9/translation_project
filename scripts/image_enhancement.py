import cv2
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

def enhance_image(image_path):
    """
    Enhances the image to improve OCR accuracy.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image could not be loaded. Check the path.")
        scale_percent = 150  # Increase size by 150%
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        enhanced_image_path = 'enhanced_image.png'
        cv2.imwrite(enhanced_image_path, img)
        return enhanced_image_path
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        raise
