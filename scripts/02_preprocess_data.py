#!/usr/bin/env python3
"""
Script 02: Preprocess VisDrone Dataset
- Process THREE SEPARATE datasets (train/val/test-dev)
- Resize images to 640x640 (with padding, not cropping)
- Convert annotations to YOLO format
- Apply light augmentation ONLY to training data
- Test data kept completely untouched
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.dataset.preprocessor import DatasetPreprocessor

def main():
    print("=" * 80)
    print("VISDRONE DATASET PREPROCESSING")
    print("=" * 80)
    
    # Get configuration
    raw_path = config.dataset_raw_path
    processed_path = config.dataset_processed_path
    target_size = tuple(config.get('preprocessing.target_size'))
    
    print(f"\nConfiguration:")
    print(f"  Raw data path: {raw_path}")
    print(f"  Processed data path: {processed_path}")
    print(f"  Target size: {target_size}")
    print(f"  Augmentation enabled: {config.get('preprocessing.augmentation.enabled', True)}")
    
    # Check if raw data exists
    if not raw_path.exists():
        print(f"\nError: Raw data directory not found: {raw_path}")
        print("Please ensure the VisDrone datasets are downloaded and placed correctly.")
        print("\nExpected structure:")
        print("  data/raw/")
        print("    ├── VisDrone2019-DET-train/")
        print("    │   ├── images/")
        print("    │   └── annotations/")
        print("    ├── VisDrone2019-DET-val/")
        print("    │   ├── images/")
        print("    │   └── annotations/")
        print("    └── VisDrone2019-DET-test-dev/")
        print("        ├── images/")
        print("        └── annotations/")
        return
    
    # Check if dataset folders exist
    datasets = [
        ('train', raw_path / 'VisDrone2019-DET-train'),
        ('val', raw_path / 'VisDrone2019-DET-val'),
        ('test-dev', raw_path / 'VisDrone2019-DET-test-dev'),
    ]
    
    missing = []
    for name, path in datasets:
        if not (path / 'images').exists() or not (path / 'annotations').exists():
            missing.append(name)
    
    if missing:
        print(f"\nError: Missing datasets: {', '.join(missing)}")
        print("Please ensure all three VisDrone datasets are downloaded.")
        return
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(raw_path, processed_path, target_size)
    
    print("\nStarting preprocessing...")
    
    # Process all three datasets separately
    stats = preprocessor.process_separate_datasets()
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    
    total_images = 0
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        print(f"  Total: {split_stats['total_images']} images")
        print(f"  Processed: {split_stats['processed']} images")
        print(f"  Failed: {split_stats['failed']} images")
        total_images += split_stats['processed']
    
    print(f"\nTotal processed: {total_images} images")
    print(f"\nProcessed data saved to: {processed_path}")
    print("\nIMPORTANT NOTES:")
    print("  ✓ Datasets processed SEPARATELY (no merging)")
    print("  ✓ Augmentation applied ONLY to train split")
    print("  ✓ Test images kept completely untouched")
    print("  ✓ All coordinates preserved in original image space")
    
    print("\nNext steps:")
    print("  1. Run script 03_run_yolo.py for YOLO inference on test set")
    print("  2. Run script 04_run_rfdetr.py for RT-DETR inference on test set")
    print("  3. Run script 05_run_sam.py for SAM segmentation on test set")
    print("  4. Run script 06_evaluate_metrics.py to evaluate")

if __name__ == "__main__":
    main()
