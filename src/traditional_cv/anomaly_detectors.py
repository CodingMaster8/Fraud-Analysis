"""
Pixel-level anomaly detection algorithms
"""
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Tuple, Dict, Any
from scipy import stats

class ELADetector:
    """Error Level Analysis - Detects JPEG compression inconsistencies"""
    
    def __init__(self, quality: int = 90, scale: int = 15):
        """
        Initialize ELA detector
        
        Args:
            quality: JPEG compression quality (0-100)
            scale: Amplification factor for differences
        """
        self.quality = quality
        self.scale = scale
    
    def analyze(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Perform Error Level Analysis on image
        
        Args:
            image: Input image (BGR format from cv2)
            
        Returns:
            ela_image: ELA visualization
            metrics: Dictionary with analysis metrics
        """
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Save as JPEG with specified quality
        buffer = BytesIO()
        pil_image.save(buffer, 'JPEG', quality=self.quality)
        buffer.seek(0)
        
        # Reload compressed image
        compressed_image = Image.open(buffer)
        compressed_array = np.array(compressed_image)
        
        # Calculate difference
        original_array = np.array(pil_image)
        ela_diff = np.abs(original_array.astype(np.float32) - compressed_array.astype(np.float32))
        
        # Scale to amplify differences
        ela_scaled = np.clip(ela_diff * self.scale, 0, 255).astype(np.uint8)
        
        # Convert back to BGR for OpenCV
        ela_bgr = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2BGR)
        
        # Calculate metrics
        ela_gray = cv2.cvtColor(ela_scaled, cv2.COLOR_RGB2GRAY)
        
        metrics = {
            "mean_error": float(np.mean(ela_gray)),
            "max_error": float(np.max(ela_gray)),
            "std_error": float(np.std(ela_gray)),
            "high_error_percentage": float(np.sum(ela_gray > 128) / ela_gray.size * 100),
            "anomaly_score": float(np.mean(ela_gray) + np.std(ela_gray))  # Combined metric
        }
        
        return ela_bgr, metrics


class NoiseDetector:
    """Noise Analysis - Detects inconsistent noise patterns across regions"""
    
    def __init__(self, patch_size: int = 64, stride: int = 32):
        """
        Initialize noise detector
        
        Args:
            patch_size: Size of patches to analyze
            stride: Stride for sliding window
        """
        self.patch_size = patch_size
        self.stride = stride
    
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level using Median Absolute Deviation (MAD)
        
        Args:
            image: Grayscale image
            
        Returns:
            Noise level estimate
        """
        # Use Laplacian to estimate noise
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        mad = np.median(np.abs(laplacian - np.median(laplacian)))
        sigma = 1.4826 * mad  # Convert MAD to sigma
        return float(sigma)
    
    def analyze(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyze noise consistency across image regions
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            noise_map: Heatmap showing noise level variations
            metrics: Dictionary with analysis metrics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate noise levels for patches
        noise_levels = []
        noise_map = np.zeros((h, w), dtype=np.float32)
        patch_count = np.zeros((h, w), dtype=np.int32)
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = gray[y:y+self.patch_size, x:x+self.patch_size]
                noise_level = self.estimate_noise_level(patch)
                noise_levels.append(noise_level)
                
                # Add to noise map
                noise_map[y:y+self.patch_size, x:x+self.patch_size] += noise_level
                patch_count[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Average noise map
        noise_map = np.divide(noise_map, patch_count, where=patch_count > 0)
        
        # Normalize for visualization
        if len(noise_levels) > 0:
            noise_levels = np.array(noise_levels)
            noise_map_normalized = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            noise_map_colored = cv2.applyColorMap(noise_map_normalized, cv2.COLORMAP_JET)
            
            # Calculate metrics
            mean_noise = float(np.mean(noise_levels))
            std_noise = float(np.std(noise_levels))
            cv_noise = std_noise / mean_noise if mean_noise > 0 else 0  # Coefficient of variation
            
            # Detect outlier regions (high inconsistency)
            z_scores = np.abs(stats.zscore(noise_levels))
            outlier_percentage = float(np.sum(z_scores > 2) / len(z_scores) * 100)
            
            metrics = {
                "mean_noise": mean_noise,
                "std_noise": std_noise,
                "coefficient_of_variation": cv_noise,
                "outlier_percentage": outlier_percentage,
                "min_noise": float(np.min(noise_levels)),
                "max_noise": float(np.max(noise_levels)),
                "anomaly_score": cv_noise * 100 + outlier_percentage  # Combined metric
            }
        else:
            noise_map_colored = np.zeros_like(image)
            metrics = {
                "mean_noise": 0.0,
                "std_noise": 0.0,
                "coefficient_of_variation": 0.0,
                "outlier_percentage": 0.0,
                "min_noise": 0.0,
                "max_noise": 0.0,
                "anomaly_score": 0.0
            }
        
        return noise_map_colored, metrics


class EdgeDetector:
    """Edge Detection - Identifies unnatural edges around altered areas"""
    
    def __init__(self, 
                 canny_low: int = 50,
                 canny_high: int = 150,
                 blur_size: int = 5):
        """
        Initialize edge detector
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            blur_size: Gaussian blur kernel size
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_size = blur_size
    
    def analyze(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyze edges to detect unnatural alterations
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            edge_composite: Composite visualization of edge analysis
            metrics: Dictionary with analysis metrics
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Calculate edge gradient strength
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_normalized = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Detect double JPEG artifacts using blocking artifacts
        # Divide into 8x8 blocks (JPEG standard)
        h, w = gray.shape
        block_diff_map = np.zeros_like(gray, dtype=np.float32)
        
        for y in range(8, h - 8, 8):
            for x in range(8, w - 8, 8):
                # Calculate difference at block boundaries
                horizontal_diff = np.abs(gray[y, x-1].astype(np.float32) - gray[y, x].astype(np.float32))
                vertical_diff = np.abs(gray[y-1, x].astype(np.float32) - gray[y, x].astype(np.float32))
                block_diff_map[y-2:y+2, x-2:x+2] = max(horizontal_diff, vertical_diff)
        
        block_diff_normalized = cv2.normalize(block_diff_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create composite visualization
        edge_colored = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
        gradient_colored = cv2.applyColorMap(gradient_normalized, cv2.COLORMAP_JET)
        block_colored = cv2.applyColorMap(block_diff_normalized, cv2.COLORMAP_VIRIDIS)
        
        # Combine visualizations
        edge_composite = cv2.addWeighted(edge_colored, 0.4, gradient_colored, 0.4, 0)
        edge_composite = cv2.addWeighted(edge_composite, 0.8, block_colored, 0.2, 0)
        
        # Calculate metrics
        edge_density = float(np.sum(edges > 0) / edges.size * 100)
        mean_gradient = float(np.mean(gradient_magnitude))
        std_gradient = float(np.std(gradient_magnitude))
        
        # Detect strong edges (potential manipulation boundaries)
        strong_edges = gradient_magnitude > (mean_gradient + 2 * std_gradient)
        strong_edge_percentage = float(np.sum(strong_edges) / strong_edges.size * 100)
        
        # Block artifact score
        block_artifact_score = float(np.mean(block_diff_map))
        
        metrics = {
            "edge_density": edge_density,
            "mean_gradient": mean_gradient,
            "std_gradient": std_gradient,
            "strong_edge_percentage": strong_edge_percentage,
            "block_artifact_score": block_artifact_score,
            "anomaly_score": strong_edge_percentage * 10 + block_artifact_score  # Combined metric
        }
        
        return edge_composite, metrics


class WhiteOutDetector:
    """Physical Document Fraud Detection - Detects white-out through brilliant white color changes"""
    
    def __init__(self, 
                 paper_white_range: Tuple[int, int] = (220, 245),
                 whiteout_white_threshold: int = 246,
                 brightness_std_threshold: float = 15.0,
                 color_uniformity_threshold: float = 10.0,
                 min_region_size: int = 100):
        """
        Initialize physical fraud detector focused on white intensity differences
        
        Args:
            paper_white_range: Expected range for normal paper white (min, max)
            whiteout_white_threshold: Threshold for detecting brilliant white (correction fluid)
            brightness_std_threshold: Max std dev for suspicious uniform regions
            color_uniformity_threshold: Max RGB channel difference for unnatural white
            min_region_size: Minimum pixel area to consider as suspicious region
        """
        self.paper_white_min = paper_white_range[0]
        self.paper_white_max = paper_white_range[1]
        self.whiteout_threshold = whiteout_white_threshold
        self.brightness_std_threshold = brightness_std_threshold
        self.color_uniformity_threshold = color_uniformity_threshold
        self.min_region_size = min_region_size
    
    def _calculate_white_intensity_map(self, gray: np.ndarray) -> np.ndarray:
        """Calculate map showing white intensity levels"""
        # Create zones: background (0-219), paper white (220-245), brilliant white (246-255)
        intensity_map = np.zeros_like(gray, dtype=np.uint8)
        intensity_map[gray >= self.paper_white_min] = 128  # Paper white
        intensity_map[gray >= self.whiteout_threshold] = 255  # Brilliant white
        return intensity_map
    
    def _detect_color_uniformity_anomalies(self, image: np.ndarray, white_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect unnatural color uniformity in white regions
        White-out has very uniform RGB values, unlike natural paper
        """
        b, g, r = cv2.split(image)
        
        # Calculate per-pixel RGB standard deviation
        rgb_stack = np.stack([r, g, b], axis=-1).astype(np.float32)
        rgb_std = np.std(rgb_stack, axis=2)
        
        # Natural paper has slight color variation, white-out is extremely uniform
        uniform_regions = (rgb_std < self.color_uniformity_threshold).astype(np.uint8) * 255
        
        # Combine with white regions
        suspicious_uniform = cv2.bitwise_and(uniform_regions, white_mask)
        
        # Calculate mean uniformity score
        mean_uniformity = float(np.mean(rgb_std[white_mask > 0])) if np.sum(white_mask > 0) > 0 else 0.0
        
        return suspicious_uniform, mean_uniformity
    
    def _detect_brightness_transitions(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect sharp transitions between normal white and brilliant white
        White-out creates unnatural brightness boundaries
        """
        # Calculate local brightness gradient
        sobelx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Focus on high-intensity regions
        high_intensity_mask = (gray > self.paper_white_min).astype(np.uint8) * 255
        
        # Detect strong gradients in white regions (paper->whiteout transitions)
        gradient_threshold = np.percentile(gradient_magnitude[high_intensity_mask > 0], 90) if np.sum(high_intensity_mask > 0) > 0 else 10
        strong_transitions = (gradient_magnitude > gradient_threshold).astype(np.uint8) * 255
        strong_transitions = cv2.bitwise_and(strong_transitions, high_intensity_mask)
        
        transition_score = float(np.sum(strong_transitions > 0) / np.sum(high_intensity_mask > 0) * 100) if np.sum(high_intensity_mask > 0) > 0 else 0.0
        
        return strong_transitions, transition_score
    
    def _analyze_local_brightness_variance(self, gray: np.ndarray, white_mask: np.ndarray, kernel_size: int = 15) -> Tuple[np.ndarray, float]:
        """
        Analyze local brightness variance - white-out has unnaturally low variance
        """
        # Calculate local standard deviation
        mean_local = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        mean_sq_local = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        variance_local = mean_sq_local - mean_local ** 2
        std_local = np.sqrt(np.maximum(variance_local, 0))
        
        # Low variance in white regions is suspicious
        low_variance_regions = (std_local < self.brightness_std_threshold).astype(np.uint8) * 255
        suspicious_low_variance = cv2.bitwise_and(low_variance_regions, white_mask)
        
        mean_variance = float(np.mean(std_local[white_mask > 0])) if np.sum(white_mask > 0) > 0 else 0.0
        
        return suspicious_low_variance, mean_variance
    
    def analyze(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Analyze document for white-out through brilliant white detection
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            composite_result: Visualization showing white intensity analysis
            metrics: Dictionary with analysis metrics
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Identify different white intensity zones
        paper_white_mask = ((gray >= self.paper_white_min) & (gray <= self.paper_white_max)).astype(np.uint8) * 255
        brilliant_white_mask = (gray >= self.whiteout_threshold).astype(np.uint8) * 255
        all_white_mask = (gray >= self.paper_white_min).astype(np.uint8) * 255
        
        # 2. Detect color uniformity anomalies
        uniform_anomalies, mean_color_uniformity = self._detect_color_uniformity_anomalies(image, brilliant_white_mask)
        
        # 3. Detect brightness transitions
        transition_regions, transition_score = self._detect_brightness_transitions(gray)
        
        # 4. Analyze local brightness variance
        low_variance_regions, mean_brightness_variance = self._analyze_local_brightness_variance(gray, brilliant_white_mask)
        
        # 5. Combine all indicators for final suspicious regions
        # Brilliant white + low color variance + low brightness variance = likely white-out
        combined_suspicious = cv2.bitwise_and(
            cv2.bitwise_and(brilliant_white_mask, uniform_anomalies),
            low_variance_regions
        )
        
        # 6. Find and analyze suspicious regions
        contours, _ = cv2.findContours(combined_suspicious, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        suspicious_regions = []
        result_overlay = image.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_region_size:
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                # Extract region for detailed analysis
                region_gray = gray[y:y+h_box, x:x+w_box]
                region_rgb = image[y:y+h_box, x:x+w_box]
                
                # Calculate region metrics
                mean_intensity = float(np.mean(region_gray))
                std_intensity = float(np.std(region_gray))
                
                # RGB uniformity
                b_std = float(np.std(region_rgb[:, :, 0]))
                g_std = float(np.std(region_rgb[:, :, 1]))
                r_std = float(np.std(region_rgb[:, :, 2]))
                rgb_uniformity = (b_std + g_std + r_std) / 3
                
                # Confidence score (higher = more likely white-out)
                confidence = (
                    (mean_intensity - self.whiteout_threshold) +  # Brightness excess
                    (self.color_uniformity_threshold - rgb_uniformity) * 2 +  # Color uniformity
                    (self.brightness_std_threshold - std_intensity)  # Low variance
                ) / 3
                
                suspicious_regions.append({
                    'x': x, 'y': y, 'w': w_box, 'h': h_box,
                    'area': area,
                    'mean_intensity': mean_intensity,
                    'brightness_std': std_intensity,
                    'rgb_uniformity': rgb_uniformity,
                    'confidence': max(0, min(100, confidence * 10))  # Scale to 0-100
                })
                
                # Draw on overlay with confidence-based color
                color = (0, 0, 255) if confidence > 5 else (0, 165, 255)  # Red = high, Orange = medium
                cv2.rectangle(result_overlay, (x, y), (x+w_box, y+h_box), color, 2)
                cv2.putText(result_overlay, f"{confidence*10:.0f}%", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 7. Create visualization
        # Intensity map showing white zones
        intensity_viz = self._calculate_white_intensity_map(gray)
        intensity_colored = cv2.applyColorMap(intensity_viz, cv2.COLORMAP_HOT)
        
        # Uniformity map
        uniformity_colored = cv2.applyColorMap(uniform_anomalies, cv2.COLORMAP_WINTER)
        
        # Transitions map
        transitions_colored = cv2.applyColorMap(transition_regions, cv2.COLORMAP_JET)
        
        # Create composite
        top_row = np.hstack([image, result_overlay])
        bottom_row = np.hstack([intensity_colored, uniformity_colored])
        composite_result = np.vstack([top_row, bottom_row])
        
        # 8. Calculate comprehensive metrics
        total_pixels = gray.size
        paper_white_percentage = float(np.sum(paper_white_mask > 0) / total_pixels * 100)
        brilliant_white_percentage = float(np.sum(brilliant_white_mask > 0) / total_pixels * 100)
        suspicious_area_percentage = float(np.sum(combined_suspicious > 0) / total_pixels * 100)
        
        # Overall anomaly score
        anomaly_score = (
            len(suspicious_regions) * 15 +  # Number of suspicious regions
            suspicious_area_percentage * 8 +  # Area coverage
            (self.color_uniformity_threshold - mean_color_uniformity) * 3 +  # Unnatural uniformity
            transition_score * 2 +  # Sharp transitions
            brilliant_white_percentage * 5  # Excessive brilliant white
        )
        
        # Determine fraud likelihood
        if anomaly_score > 150 or len(suspicious_regions) >= 3:
            fraud_likelihood = "HIGH"
        elif anomaly_score > 75 or len(suspicious_regions) >= 1:
            fraud_likelihood = "MEDIUM"
        else:
            fraud_likelihood = "LOW"
        
        metrics = {
            "paper_white_percentage": paper_white_percentage,
            "brilliant_white_percentage": brilliant_white_percentage,
            "white_intensity_contrast": brilliant_white_percentage / max(paper_white_percentage, 0.1),
            "suspicious_region_count": len(suspicious_regions),
            "suspicious_area_percentage": suspicious_area_percentage,
            "suspicious_regions": suspicious_regions,
            "mean_color_uniformity": mean_color_uniformity,
            "mean_brightness_variance": mean_brightness_variance,
            "transition_score": transition_score,
            "anomaly_score": float(anomaly_score),
            "fraud_likelihood": fraud_likelihood
        }
        
        return composite_result, metrics