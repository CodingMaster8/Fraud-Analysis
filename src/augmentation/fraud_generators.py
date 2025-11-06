"""
Fraud generation techniques for creating artificial fraudulent checks
"""
import cv2
import numpy as np


class GeometricAugmenter:
    """Apply geometric transformations and noise"""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle (degrees)"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), 
                             borderMode=cv2.BORDER_CONSTANT, 
                             borderValue=(255, 255, 255))
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to image"""
        noise = np.random.normal(0, noise_level * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Adjust brightness by factor"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """Adjust contrast by factor"""
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    @staticmethod
    def scale(image: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale image uniformly"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        scaled = cv2.resize(image, (new_w, new_h))
        
        # Pad or crop to original size
        if scale_factor < 1:
            # Pad with white
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            scaled = cv2.copyMakeBorder(scaled, pad_h, h - new_h - pad_h, 
                                       pad_w, w - new_w - pad_w,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            # Crop from center
            start_h, start_w = (new_h - h) // 2, (new_w - w) // 2
            scaled = scaled[start_h:start_h+h, start_w:start_w+w]
        
        return scaled
