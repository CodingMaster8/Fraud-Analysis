"""
Apply geometric transformations to existing fraudulent check images
"""
import cv2
import numpy as np
import json
from tqdm import tqdm
from pydantic import BaseModel, Field
from typing import Tuple, List, Dict, Any

from .config import DATA_DIR, OUTPUT_DIR
from .fraud_generators import GeometricAugmenter

class AugmentationConfig(BaseModel):
    """Configuration for geometric augmentations"""
    rotation_range: Tuple[float, float] = Field(default=(-10, 10))
    scale_range: Tuple[float, float] = Field(default=(0.9, 1.1))
    noise_level: Tuple[float, float] = Field(default=(0.01, 0.05))
    brightness_range: Tuple[float, float] = Field(default=(0.8, 1.2))
    contrast_range: Tuple[float, float] = Field(default=(0.8, 1.2))
    
    @classmethod
    async def create(cls, **kwargs):
        """Asynchronous factory method to create an instance"""
        instance = cls(**kwargs)
        return instance


class GeometricOnlyPipeline(BaseModel):
    """Pipeline for applying only geometric transformations to existing images"""
    augmentation_config: AugmentationConfig = Field(default_factory=AugmentationConfig, description="Configuration for geometric augmentations")
    metadata: List[Dict[str, Any]] = Field(default_factory=list)
    
    model_config = {
        "arbitrary_types_allowed": True
    }

    def apply_geometric_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply geometric transformations and noise"""
        
        # Rotation
        if np.random.random() < 0.7:
            angle = np.random.uniform(*self.augmentation_config.rotation_range)
            image = GeometricAugmenter.rotate(image, angle)
        
        # Scale
        if np.random.random() < 0.5:
            scale = np.random.uniform(*self.augmentation_config.scale_range)
            image = GeometricAugmenter.scale(image, scale)
        
        # Noise
        if np.random.random() < 0.6:
            noise_level = np.random.uniform(*self.augmentation_config.noise_level)
            image = GeometricAugmenter.add_noise(image, noise_level)
        
        # Brightness
        if np.random.random() < 0.6:
            brightness = np.random.uniform(*self.augmentation_config.brightness_range)
            image = GeometricAugmenter.adjust_brightness(image, brightness)
        
        # Contrast
        if np.random.random() < 0.5:
            contrast = np.random.uniform(*self.augmentation_config.contrast_range)
            image = GeometricAugmenter.adjust_contrast(image, contrast)
        
        return image
    
    def process_dataset(self, versions_per_image: int = 5):
        """
        Process existing fraudulent images
        
        Args:
            versions_per_image: Number of augmented versions per image
        """
        # Create output directories
        (OUTPUT_DIR / "fraudulent").mkdir(parents=True, exist_ok=True)
        
        # Process fraudulent images
        fraud_dir = DATA_DIR / "fraudulent"
        fraud_files = list(fraud_dir.glob("*.jpg")) + \
                      list(fraud_dir.glob("*.png")) + \
                      list(fraud_dir.glob("*.jpeg"))
        
        print(f"Found {len(fraud_files)} fraudulent check images")
        print(f"Generating {versions_per_image} augmented versions per image...")
        
        fraud_count = 0
        
        for img_path in tqdm(fraud_files, desc="Processing fraudulent images"):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read {img_path}")
                continue
            
            # Copy original to output
            original_output = OUTPUT_DIR / "fraudulent" / f"original_{img_path.name}"
            cv2.imwrite(str(original_output), image)
            
            self.metadata.append({
                "filename": f"original_{img_path.name}",
                "original_file": img_path.name,
                "label": "fraudulent",
                "augmentation_type": "none",
                "is_original": True
            })
            fraud_count += 1
            
            # Generate augmented versions
            for i in range(versions_per_image):
                aug_image = self.apply_geometric_augmentation(image.copy())
                
                output_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
                output_path = OUTPUT_DIR / "fraudulent" / output_name
                cv2.imwrite(str(output_path), aug_image)
                
                self.metadata.append({
                    "filename": output_name,
                    "original_file": img_path.name,
                    "label": "fraudulent",
                    "augmentation_type": "geometric_only",
                    "is_original": False
                })
                fraud_count += 1
        
        # Save metadata
        metadata_path = OUTPUT_DIR / "augmentation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print("✓ Dataset augmentation complete!")
        print(f"{'='*60}")
        print(f"Fraudulent images:")
        print(f"  - Original: {len(fraud_files)}")
        print(f"  - Augmented: {fraud_count - len(fraud_files)}")
        print(f"  - Total: {fraud_count}")
        print(f"\nMetadata saved to: {metadata_path}")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"{'='*60}")
