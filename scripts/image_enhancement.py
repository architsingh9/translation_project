import cv2
import os
import logging

logger = logging.getLogger(__name__)

def enhance_image(image_path):
    """
    Enhances the image to improve OCR accuracy.
    """
    try:
        # Load the image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Upscale the image
        scale_percent = 150  # Increase size by 150%
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

        # Apply Gaussian Blur to reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # Sharpen the image
        import numpy as np
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)

        # Save the enhanced image temporarily
        enhanced_image_path = 'enhanced_image.png'
        cv2.imwrite(enhanced_image_path, img)
        return enhanced_image_path
    except Exception as e:
        logger.error(f"Image enhancement failed: {e}")
        raise
