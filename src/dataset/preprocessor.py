import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import shutil
from tqdm import tqdm
import random
import json

from src.config import config
from src.dataset.parser import VisDroneParser

class DatasetPreprocessor:
    """Preprocess VisDrone dataset for model inference.
    
    IMPORTANT: This preprocessor handles THREE SEPARATE VisDrone datasets:
    - VisDrone2019-DET-train (~7145 images)
    - VisDrone2019-DET-val (~1580 images)  
    - VisDrone2019-DET-test-dev (~1610 images)
    
    These are processed SEPARATELY to maintain dataset integrity and avoid data leakage.
    """
    
    def __init__(self, raw_path: str, processed_path: str, target_size: Tuple[int, int] = (640, 640)):
        self.raw_path = Path(raw_path)
        self.processed_path = Path(processed_path)
        self.target_size = target_size
        self.stats = {'train': {}, 'val': {}, 'test': {}}
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            (self.processed_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.processed_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image while maintaining aspect ratio with padding"""
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate padding offsets (center the image)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply light augmentation to image"""
        # Random horizontal flip
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random brightness adjustment
        brightness = random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
        
        return image
    
    def process_split(self, split_name: str, images_dir: str, annotations_dir: str, apply_augmentation: bool = False):
        """
        Process a single VisDrone split (train/val/test-dev).
        Does NOT augment test data by design.
        
        Args:
            split_name: 'train', 'val', or 'test'
            images_dir: Directory with this split's images
            annotations_dir: Directory with this split's annotations  
            apply_augmentation: Whether to apply augmentation (ONLY for train split)
        """
        images_path = Path(images_dir)
        image_files = sorted(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png')))
        
        if len(image_files) == 0:
            print(f"Warning: Found no images in {images_dir}")
            return
        
        print(f"\nProcessing {split_name.upper()} split ({len(image_files)} images)...")
        
        # Initialize parser
        parser = VisDroneParser(annotations_dir)
        
        image_ids = []
        processed_count = 0
        failed_count = 0
        
        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read {img_path}")
                failed_count += 1
                continue
            
            orig_h, orig_w = image.shape[:2]
            image_id = img_path.stem
            image_ids.append(image_id)
            
            # Apply augmentation ONLY to training data
            if split_name == 'train' and apply_augmentation:
                image = self.augment_image(image)
            
            # Resize image (with padding, not cropping)
            processed_image = self.resize_image(image)
            
            # Save processed image
            output_img_path = self.processed_path / split_name / 'images' / f"{image_id}.jpg"
            cv2.imwrite(str(output_img_path), processed_image)
            
            # Parse and convert annotations
            ann_file = Path(annotations_dir) / f"{image_id}.txt"
            if ann_file.exists():
                annotations = parser.parse_annotation(str(ann_file))
                # IMPORTANT: Keep original dimensions for ground truth
                # (they will be used as-is for evaluation)
                yolo_annotations = parser.convert_to_yolo_format(annotations, orig_w, orig_h)
                
                # Save annotations
                output_ann_path = self.processed_path / split_name / 'labels' / f"{image_id}.txt"
                with open(output_ann_path, 'w') as f:
                    f.write('\n'.join(yolo_annotations))
            
            processed_count += 1
        
        self.stats[split_name] = {
            'total_images': len(image_files),
            'processed': processed_count,
            'failed': failed_count
        }
        
        # Save split info for reference
        split_file = config.splits_path / f"{split_name}.txt"
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, 'w') as f:
            for img_id in image_ids:
                f.write(f"{img_id}\n")
        
        print(f"✓ {split_name.upper()} split: {processed_count} images processed, {failed_count} failed")
        
        return image_ids
    
    def process_separate_datasets(self):
        """
        Process all three VisDrone splits from SEPARATE folders.
        
        Expected structure:
            data/raw/
            ├── VisDrone2019-DET-train/
            │   ├── images/
            │   └── annotations/
            ├── VisDrone2019-DET-val/
            │   ├── images/
            │   └── annotations/
            └── VisDrone2019-DET-test-dev/
                ├── images/
                └── annotations/
        """
        
        datasets = [
            ('train', self.raw_path / 'VisDrone2019-DET-train'),
            ('val', self.raw_path / 'VisDrone2019-DET-val'),
            ('test', self.raw_path / 'VisDrone2019-DET-test-dev'),
        ]
        
        all_stats = {}
        
        for split_name, dataset_path in datasets:
            images_dir = dataset_path / 'images'
            annotations_dir = dataset_path / 'annotations'
            
            if not images_dir.exists() or not annotations_dir.exists():
                print(f"Error: Cannot find images/annotations for {split_name} split")
                print(f"  Expected: {images_dir}")
                print(f"  And:      {annotations_dir}")
                continue
            
            # Augment only training data
            apply_aug = (split_name == 'train') and config.get('preprocessing.augmentation.enabled', True)
            self.process_split(split_name, str(images_dir), str(annotations_dir), apply_augmentation=apply_aug)
            all_stats[split_name] = self.stats[split_name]
        
        return all_stats
    

def main():
    """Main preprocessing function"""
    from src.config import config
    
    # Get paths from config
    raw_path = config.dataset_raw_path
    processed_path = config.dataset_processed_path
    target_size = tuple(config.get('preprocessing.target_size'))
    
    print(f"Raw path: {raw_path}")
    print(f"Processed path: {processed_path}")
    
    if not raw_path.exists():
        print(f"Error: Raw data directory not found: {raw_path}")
        return
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(raw_path, processed_path, target_size)
    
    # Process all three separate datasets
    print("\n" + "="*80)
    print("PREPROCESSING VISDRONE DATASETS (SEPARATE SPLITS)")
    print("="*80)
    
    stats = preprocessor.process_separate_datasets()
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    
    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()}:")
        for key, value in split_stats.items():
            print(f"  {key}: {value}")
    
    print(f"\nProcessed data saved to: {processed_path}")

if __name__ == "__main__":
    main()
