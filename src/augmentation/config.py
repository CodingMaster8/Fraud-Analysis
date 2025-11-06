"""
Configuration for data augmentation pipeline
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NON_FRAUD_DIR = DATA_DIR / "non_fraud"
FRAUDULENT_DIR = DATA_DIR / "fraudulent"
OUTPUT_DIR = DATA_DIR / "augmented"

# Output directories
(OUTPUT_DIR / "fraudulent").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "non_fraud").mkdir(parents=True, exist_ok=True)
