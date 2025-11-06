"""
Main analyzer pipeline for pixel-level anomaly detection
"""
import cv2
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Union, Tuple
import json

from .anomaly_detectors import ELADetector, NoiseDetector, EdgeDetector, WhiteOutDetector
from .config import OUTPUT_DIR


class AnalysisConfig(BaseModel):
    """Configuration for anomaly analysis"""
    enable_ela: bool = Field(default=True, description="Enable Error Level Analysis")
    enable_noise: bool = Field(default=True, description="Enable Noise Analysis")
    enable_edge: bool = Field(default=True, description="Enable Edge Detection")
    enable_whiteout: bool = Field(default=True, description="Enable Whiteout Fraud Detection")
    
    ela_quality: int = Field(default=90, description="JPEG quality for ELA")
    ela_scale: int = Field(default=15, description="ELA amplification scale")
    
    noise_patch_size: int = Field(default=64, description="Noise analysis patch size")
    noise_stride: int = Field(default=32, description="Noise analysis stride")
    
    edge_canny_low: int = Field(default=50, description="Canny lower threshold")
    edge_canny_high: int = Field(default=150, description="Canny upper threshold")
    edge_blur_size: int = Field(default=5, description="Gaussian blur kernel size")

    paper_white_range: Tuple[int, int] = Field(default=(220, 245), description="Expected range for normal paper white (min, max)")
    whiteout_white_threshold: int = Field(default=246, description="Threshold for detecting brilliant white (correction fluid)")
    brightness_std_threshold: float = Field(default=15.0, description="Max std dev for suspicious uniform regions")
    color_uniformity_threshold: float = Field(default=10.0, description="Max std dev for suspicious uniform regions")
    min_region_size: int = Field(default=100, description="Minimum pixel area to consider as suspicious region")
    
    save_visualizations: bool = Field(default=True, description="Save visualization images")
    
    @classmethod
    async def create(cls, **kwargs):
        """Asynchronous factory method to create an instance"""
        return cls(**kwargs)


class AnalysisResult(BaseModel):
    """Results from anomaly detection analysis"""
    image_path: str
    ela_metrics: Optional[Dict[str, Any]] = None
    noise_metrics: Optional[Dict[str, Any]] = None
    edge_metrics: Optional[Dict[str, Any]] = None
    whiteout_metrics: Optional[Dict[str, Any]] = None
    overall_score: float = Field(default=0.0, description="Combined anomaly score")
    fraud_likelihood: str = Field(default="Unknown", description="Low/Medium/High/Very High")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "image_path": self.image_path,
            "ela_metrics": self.ela_metrics,
            "noise_metrics": self.noise_metrics,
            "edge_metrics": self.edge_metrics,
            "overall_score": self.overall_score,
            "fraud_likelihood": self.fraud_likelihood
        }


class AnomalyAnalyzer(BaseModel):
    """Main pipeline for analyzing images for pixel-level anomalies"""
    config: AnalysisConfig = Field(default_factory=AnalysisConfig)
    ela_detector: Optional[ELADetector] = Field(default=None, exclude=True)
    noise_detector: Optional[NoiseDetector] = Field(default=None, exclude=True)
    edge_detector: Optional[EdgeDetector] = Field(default=None, exclude=True)
    whiteout_detector: Optional[WhiteOutDetector] = Field(default=None, exclude=True)
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    def _initialize_detectors(self):
        """Initialize detector instances"""
        
        if self.config.enable_ela:
            self.ela_detector = ELADetector(
                quality=self.config.ela_quality,
                scale=self.config.ela_scale
            )
        
        if self.config.enable_noise:
            self.noise_detector = NoiseDetector(
                patch_size=self.config.noise_patch_size,
                stride=self.config.noise_stride
            )
        
        if self.config.enable_edge:
            self.edge_detector = EdgeDetector(
                canny_low=self.config.edge_canny_low,
                canny_high=self.config.edge_canny_high,
                blur_size=self.config.edge_blur_size
            )
        if self.config.enable_whiteout:
            self.whiteout_detector = WhiteOutDetector(
                paper_white_range=self.config.paper_white_range,
                whiteout_white_threshold=self.config.whiteout_white_threshold,
                brightness_std_threshold=self.config.brightness_std_threshold,
                color_uniformity_threshold=self.config.color_uniformity_threshold,
                min_region_size=self.config.min_region_size
            )
    
    def analyze_image(self, image_path: Union[str, Path]) -> AnalysisResult:
        """
        Analyze a single image for anomalies
        
        Args:
            image_path: Path to the image file
            
        Returns:
            AnalysisResult with metrics and scores
        """
        # Initialize detectors
        self._initialize_detectors()
        
        # Load image
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Initialize result
        result = AnalysisResult(image_path=str(image_path))
        
        # Create output directory for this image
        if self.config.save_visualizations:
            output_subdir = OUTPUT_DIR / image_path.stem
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Save original image
            cv2.imwrite(str(output_subdir / "original.jpg"), image)
        
        # Run ELA
        if self.config.enable_ela and self.ela_detector:
            ela_image, ela_metrics = self.ela_detector.analyze(image)
            result.ela_metrics = ela_metrics
            
            if self.config.save_visualizations:
                cv2.imwrite(str(output_subdir / "ela_analysis.jpg"), ela_image)
        
        # Run Noise Analysis
        if self.config.enable_noise and self.noise_detector:
            noise_map, noise_metrics = self.noise_detector.analyze(image)
            result.noise_metrics = noise_metrics
            
            if self.config.save_visualizations:
                cv2.imwrite(str(output_subdir / "noise_analysis.jpg"), noise_map)
        
        # Run Edge Detection
        if self.config.enable_edge and self.edge_detector:
            edge_composite, edge_metrics = self.edge_detector.analyze(image)
            result.edge_metrics = edge_metrics
            
            if self.config.save_visualizations:
                cv2.imwrite(str(output_subdir / "edge_analysis.jpg"), edge_composite)
        
        if self.config.enable_whiteout and self.whiteout_detector:
            whiteout_image, whiteout_metrics = self.whiteout_detector.analyze(image)
            result.edge_metrics = whiteout_metrics
            
            if self.config.save_visualizations:
                cv2.imwrite(str(output_subdir / "whiteout_analysis.jpg"), whiteout_image)

        # Calculate overall score
        result.overall_score = self._calculate_overall_score(result)
        result.fraud_likelihood = self._get_fraud_likelihood(result.overall_score)
        
        # Save results JSON
        if self.config.save_visualizations:
            with open(output_subdir / "analysis_results.json", 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
        
        return result
    
    def _calculate_overall_score(self, result: AnalysisResult) -> float:
        """
        Calculate overall anomaly score from individual metrics
        
        Combines scores from ELA, Noise, and Edge detection
        Higher score = more likely to be fraudulent
        """
        scores = []
        weights = []
        
        if result.ela_metrics:
            # ELA anomaly score (normalized)
            ela_score = result.ela_metrics.get("anomaly_score", 0)
            scores.append(min(ela_score / 50, 100))  # Normalize to 0-100
            weights.append(1.5)  # ELA is important
        
        if result.noise_metrics:
            # Noise inconsistency score (normalized)
            noise_score = result.noise_metrics.get("anomaly_score", 0)
            scores.append(min(noise_score, 100))  # Already in good range
            weights.append(1.2)  # Noise analysis is moderately important
        
        if result.edge_metrics:
            # Edge anomaly score (normalized)
            edge_score = result.edge_metrics.get("anomaly_score", 0)
            scores.append(min(edge_score / 2, 100))  # Normalize to 0-100
            weights.append(1.0)  # Edge detection is supporting evidence
        
        if result.whiteout_metrics:
            phys_fraud_score = result.whiteout_metrics.get("anomaly_score", 0)
            scores.append(min(phys_fraud_score, 100))  # Already in good range
            weights.append(1.3)  # Physical fraud detection is important
        
        if len(scores) == 0:
            return 0.0
        
        # Weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        return round(weighted_score, 2)
    
    def _get_fraud_likelihood(self, score: float) -> str:
        """
        Convert numeric score to categorical fraud likelihood
        
        Args:
            score: Overall anomaly score (0-100)
            
        Returns:
            Fraud likelihood category
        """
        if score < 20:
            return "Low"
        elif score < 40:
            return "Medium"
        elif score < 60:
            return "High"
        else:
            return "Very High"
    
    def analyze_batch(self, image_paths: list[Union[str, Path]]) -> list[AnalysisResult]:
        """
        Analyze multiple images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of AnalysisResult objects
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.analyze_image(image_path)
                results.append(result)
                print(f"✓ Analyzed: {image_path} - Score: {result.overall_score} ({result.fraud_likelihood})")
            except Exception as e:
                print(f"✗ Error analyzing {image_path}: {str(e)}")
        
        return results
    
    def analyze_directory(self, directory: Union[str, Path], pattern: str = "*.[jp][pn]g") -> list[AnalysisResult]:
        """
        Analyze all images in a directory
        
        Args:
            directory: Path to directory containing images
            pattern: Glob pattern for image files
            
        Returns:
            List of AnalysisResult objects
        """
        directory = Path(directory)
        image_paths = list(directory.glob(pattern))
        
        print(f"Found {len(image_paths)} images in {directory}")
        return self.analyze_batch(image_paths)
