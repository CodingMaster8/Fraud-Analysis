"""
Utility to visualize augmentation results
"""
import cv2
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import json

from .config import OUTPUT_DIR, DATA_DIR


def visualize_sample_augmentations(num_samples: int = 6):
    """
    Create a visualization grid showing original and augmented versions
    
    Args:
        num_samples: Number of samples to display
    """
    # Load metadata
    metadata_path = OUTPUT_DIR / "augmentation_metadata.json"
    
    if not metadata_path.exists():
        print("No augmentation metadata found. Run the pipeline first.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get original fraudulent images
    original_fraud = [item for item in metadata if item["label"] == "fraudulent" and item.get("is_original", False)]
    
    # Sample random original fraudulent images
    samples = random.sample(original_fraud, min(num_samples, len(original_fraud)))
    
    # Pre-load valid image pairs
    valid_pairs = []
    for sample in samples:
        # Load original image
        original_path = OUTPUT_DIR / "fraudulent" / sample["filename"]
        
        # Find an augmented version of this image
        augmented = [item for item in metadata 
                    if item["original_file"] == sample["original_file"] 
                    and not item.get("is_original", False)]
        
        if not augmented:
            continue
            
        aug_sample = random.choice(augmented)
        aug_path = OUTPUT_DIR / "fraudulent" / aug_sample["filename"]
        
        original = cv2.imread(str(original_path))
        augmented_img = cv2.imread(str(aug_path))
        
        if original is None or augmented_img is None:
            continue
        
        # Convert BGR to RGB for display
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
        
        valid_pairs.append({
            'original': original,
            'augmented': augmented_img,
            'original_name': sample["original_file"],
            'aug_type': aug_sample.get("augmentation_type", "unknown")
        })
        
        if len(valid_pairs) >= num_samples:
            break
    
    if not valid_pairs:
        print("No valid image pairs found.")
        return
    
    # Create figure with actual number of valid pairs
    actual_samples = len(valid_pairs)
    rows = (actual_samples + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(20, 6 * rows))
    fig.suptitle('Check Augmentation Results: Original vs Augmented', fontsize=16)
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, pair in enumerate(valid_pairs):
        row = idx // 2
        col_offset = (idx % 2) * 2
        
        # Display original
        axes[row, col_offset].imshow(pair['original'])
        axes[row, col_offset].set_title(f'Original: {pair["original_name"][:20]}...')
        axes[row, col_offset].axis('off')
        
        # Display augmented
        axes[row, col_offset + 1].imshow(pair['augmented'])
        axes[row, col_offset + 1].set_title(f'Augmented: {pair["aug_type"]}')
        axes[row, col_offset + 1].axis('off')
    
    # Hide unused subplots if odd number of pairs
    if actual_samples % 2 == 1:
        axes[rows - 1, 2].axis('off')
        axes[rows - 1, 3].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = OUTPUT_DIR / "augmentation_samples.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    plt.show()


def augmentation_stats():
    """Print statistics about the augmented dataset"""
    metadata_path = OUTPUT_DIR / "augmentation_metadata.json"
    
    if not metadata_path.exists():
        print("No augmentation metadata found. Run the pipeline first.")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Calculate statistics
    fraud_items = [item for item in metadata if item["label"] == "fraudulent"]
    original_fraud = [item for item in fraud_items if item.get("is_original", False)]
    augmented_fraud = [item for item in fraud_items if not item.get("is_original", False)]
    
    augmentation_types = {}
    
    for item in augmented_fraud:
        aug_type = item.get("augmentation_type", "unknown")
        augmentation_types[aug_type] = augmentation_types.get(aug_type, 0) + 1
    
    print("\n" + "="*60)
    print("AUGMENTATION STATISTICS")
    print("="*60)
    print(f"\nTotal Fraudulent Images: {len(fraud_items)}")
    print(f"  - Original: {len(original_fraud)}")
    print(f"  - Augmented: {len(augmented_fraud)}")
    
    if augmentation_types:
        print(f"\nAugmentation Types Distribution:")
        for aug_type, count in sorted(augmentation_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {aug_type}: {count} ({count/len(augmented_fraud)*100:.1f}%)")
    
    print("\n" + "="*60)