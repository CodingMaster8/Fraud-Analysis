# Traditional CV - Pixel-Level Anomaly Detection

This module implements traditional computer vision techniques to detect pixel-level anomalies in check images that may indicate fraud.

## Detection Methods

### 1. Error Level Analysis (ELA)
Detects JPEG compression inconsistencies by re-saving the image at a known quality level and analyzing the differences. Altered regions will show different compression artifacts than the rest of the image.

**Key Metrics:**
- `mean_error`: Average error level across the image
- `max_error`: Maximum error detected
- `std_error`: Standard deviation of errors
- `high_error_percentage`: Percentage of pixels with high error
- `anomaly_score`: Combined metric for fraud detection

### 2. Noise Analysis
Analyzes noise consistency across different regions of the image. Authentic images should have uniform noise patterns, while manipulated regions may have different noise characteristics.

**Key Metrics:**
- `mean_noise`: Average noise level
- `coefficient_of_variation`: Noise consistency measure
- `outlier_percentage`: Percentage of regions with unusual noise
- `anomaly_score`: Combined metric for fraud detection

### 3. Edge Detection
Identifies unnatural edges that may indicate manipulation boundaries, including:
- Sharp discontinuities from copy-paste operations
- JPEG blocking artifacts from double compression
- Gradient inconsistencies at alteration boundaries

**Key Metrics:**
- `edge_density`: Overall edge concentration
- `strong_edge_percentage`: Percentage of unusually strong edges
- `block_artifact_score`: JPEG blocking artifact detection
- `anomaly_score`: Combined metric for fraud detection

## Usage

### Basic Usage

```python
from src.traditional_cv import AnomalyAnalyzer, AnalysisConfig

# Create analyzer with default settings
analyzer = AnomalyAnalyzer()

# Analyze a single image
result = analyzer.analyze_image("path/to/check.jpg")

print(f"Overall Score: {result.overall_score}")
print(f"Fraud Likelihood: {result.fraud_likelihood}")
print(f"ELA Metrics: {result.ela_metrics}")
print(f"Noise Metrics: {result.noise_metrics}")
print(f"Edge Metrics: {result.edge_metrics}")
```

### Custom Configuration

```python
from src.traditional_cv import AnomalyAnalyzer, AnalysisConfig

# Create custom configuration
config = AnalysisConfig(
    enable_ela=True,
    enable_noise=True,
    enable_edge=True,
    ela_quality=85,
    ela_scale=20,
    noise_patch_size=128,
    save_visualizations=True
)

analyzer = AnomalyAnalyzer(config=config)
result = analyzer.analyze_image("path/to/check.jpg")
```

### Batch Analysis

```python
# Analyze multiple images
image_paths = ["check1.jpg", "check2.jpg", "check3.jpg"]
results = analyzer.analyze_batch(image_paths)

# Analyze all images in a directory
results = analyzer.analyze_directory("data/fraudulent")
```

### Using Individual Detectors

```python
from src.traditional_cv import ELADetector, NoiseDetector, EdgeDetector
import cv2

image = cv2.imread("check.jpg")

# ELA only
ela_detector = ELADetector(quality=90, scale=15)
ela_image, ela_metrics = ela_detector.analyze(image)

# Noise analysis only
noise_detector = NoiseDetector(patch_size=64, stride=32)
noise_map, noise_metrics = noise_detector.analyze(image)

# Edge detection only
edge_detector = EdgeDetector(canny_low=50, canny_high=150)
edge_composite, edge_metrics = edge_detector.analyze(image)
```

## Output

When `save_visualizations=True`, the analyzer creates:
```
output/traditional_cv/{image_name}/
    ├── original.jpg              # Original image
    ├── ela_analysis.jpg          # ELA visualization
    ├── noise_analysis.jpg        # Noise heatmap
    ├── edge_analysis.jpg         # Edge detection composite
    └── analysis_results.json     # Complete metrics
```

## Interpreting Results

### Overall Score (0-100)
- **0-20**: Low risk (likely authentic)
- **20-40**: Medium risk (requires review)
- **40-60**: High risk (likely fraudulent)
- **60+**: Very High risk (very likely fraudulent)

### ELA Interpretation
- High `mean_error` and `std_error` indicate multiple alterations
- Localized bright spots in ELA image show manipulated regions
- Uniform low error suggests authentic image

### Noise Interpretation
- High `coefficient_of_variation` indicates inconsistent noise
- High `outlier_percentage` suggests region-specific manipulation
- Colored patches in heatmap show areas with different noise

### Edge Interpretation
- High `strong_edge_percentage` may indicate copy-paste boundaries
- High `block_artifact_score` suggests double JPEG compression
- Concentrated edges in visualization may outline altered areas

## Configuration Parameters

### AnalysisConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_ela` | True | Enable Error Level Analysis |
| `enable_noise` | True | Enable Noise Analysis |
| `enable_edge` | True | Enable Edge Detection |
| `ela_quality` | 90 | JPEG quality for ELA (0-100) |
| `ela_scale` | 15 | ELA difference amplification |
| `noise_patch_size` | 64 | Size of patches for noise analysis |
| `noise_stride` | 32 | Stride for sliding window |
| `edge_canny_low` | 50 | Canny lower threshold |
| `edge_canny_high` | 150 | Canny upper threshold |
| `edge_blur_size` | 5 | Gaussian blur kernel size |
| `save_visualizations` | True | Save analysis images |
