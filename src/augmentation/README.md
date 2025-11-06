# Dummy Data Generation Pipeline

This module generates augmented fraudulent check images from existing fraudulent check images using geometric transformation techniques.

## Overview

The pipeline applies realistic geometric augmentations to existing fraudulent checks:
- **Rotation**: Random rotation within configurable angle range
- **Scaling**: Uniform scaling with padding/cropping
- **Noise**: Gaussian noise addition
- **Brightness Adjustment**: HSV-based brightness variation
- **Contrast Adjustment**: Mean-centered contrast modification

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

Organize your images in the following structure:
```
data/
  fraudulent/           # Your existing fraudulent check images
    fraud_check_1.jpg
    fraud_check_2.jpg
    ...
```

### 3. Augment Existing Fraud Images

Apply geometric transformations to your existing fraudulent images:

```bash
cd ../isitreal-pablo/src/dummy_data
python augment_existing_fraud.py --versions 5
```

**Command-line arguments:**
- `--versions`: Number of augmented versions per image (default: 5)

This will:
- Read all images from `data/fraudulent/`
- Copy original images to output directory
- Generate 5 augmented versions per image with randomized transformations
- Save originals + augmented versions to `data/augmented/fraudulent/`
- Create metadata JSON file tracking all transformations

### 4. Visualize Results (Optional)

To inspect the augmented images:

```bash
python visualize_augmentations.py
```

This will:
- Display sample augmentations
- Print statistics about generated dataset
- Save visualization grid to `data/augmented/augmentation_samples.png`

## Configuration

Edit the `AugmentationConfig` class in `augment_existing_fraud.py` to customize transformation ranges:

```python
class AugmentationConfig(BaseModel):
    rotation_range: Tuple[float, float] = (-10, 10)      # degrees
    scale_range: Tuple[float, float] = (0.9, 1.1)        # scale factor
    noise_level: Tuple[float, float] = (0.01, 0.05)      # noise intensity
    brightness_range: Tuple[float, float] = (0.8, 1.2)   # brightness factor
    contrast_range: Tuple[float, float] = (0.8, 1.2)     # contrast factor
```

**Transformation probabilities** (in `apply_geometric_augmentation`):
- Rotation: 70%
- Scaling: 50%
- Noise: 60%
- Brightness: 60%
- Contrast: 50%

## Output Structure

```
data/
  augmented/
    fraudulent/                      # All augmented fraudulent checks
      original_fraud_check_1.jpg     # Original copy
      fraud_check_1_aug0.jpg         # Augmented version 0
      fraud_check_1_aug1.jpg         # Augmented version 1
      ...
    augmentation_metadata.json       # Details about each generated image
```