"""
Configuration for traditional CV anomaly detection
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "traditional_cv"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
