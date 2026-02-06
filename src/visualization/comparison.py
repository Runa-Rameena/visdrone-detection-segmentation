#!/usr/bin/env python3
"""
Comparison Visualization Script
Create side-by-side comparisons of YOLO, RT-DETR, and SAM outputs
"""

import sys
import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.visualization.bbox_visualizer import BBoxVisualizer
from src.visualization.mask_visualizer import MaskVisualizer

def create_sample_comparisons(num_samples: int = 5):
    """Create comparison visualizations for sample images"""
    
    # Load results
    yolo_results_file = config.metrics_path / 'yolo_results.json'
    rtdetr_results_file = config.metrics_path / 'rtdetr_results.json'
    sam_results_file = config.metrics_path / 'sam_results.json'
    
    if not all([yolo_results_file.exists(), rtdetr_results_file.exists()]):
        print("Error: Detection results not found")
        return
    
    # Load data
    with open(yolo_results_file, 'r') as f:
        yolo_data = json.load(f)
    with open(rtdetr_results_file, 'r') as f:
        rtdetr_data = json.load(f)
    
    sam_data = None
    if sam_results_file.exists():
        with open(sam_results_file, 'r') as f:
            sam_data = json.load(f)
    
    yolo_detections = yolo_data['detections']
    rtdetr_detections = rtdetr_data['detections']
    
    # Initialize visualizers
    class_names = config.get('classes', {})
    bbox_viz = BBoxVisualizer(class_names)
    mask_viz = MaskVisualizer()
    
    # Create output directory
    output_dir = config.visualizations_path / 'comparisons'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_samples} sample comparisons...")
    
    # Select diverse samples (different detection counts)
    samples = []
    for i, (yolo_res, rtdetr_res) in enumerate(zip(yolo_detections, rtdetr_detections)):
        detection_count = yolo_res['num_detections'] + rtdetr_res['num_detections']
        samples.append((i, detection_count, yolo_res, rtdetr_res))
    
    # Sort by detection count and pick diverse samples
    samples.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [0]  # Highest detections
    if len(samples) > 1:
        selected_indices.append(len(samples) // 2)  # Medium
    if len(samples) > 2:
        selected_indices.append(len(samples) - 1)  # Lowest
    
    # Add random samples if needed
    import random
    random.seed(42)
    while len(selected_indices) < min(num_samples, len(samples)):
        idx = random.randint(0, len(samples) - 1)
        if idx not in selected_indices:
            selected_indices.append(idx)
    
    # Create comparisons
    for idx in selected_indices[:num_samples]:
        _, _, yolo_res, rtdetr_res = samples[idx]
        
        image_path = yolo_res['image_path']
        image_name = Path(image_path).stem
        
        print(f"Processing {image_name}...")
        
        # Create figure
        has_sam = sam_data is not None
        num_cols = 4 if has_sam else 3
        fig, axes = plt.subplots(1, num_cols, figsize=(6 * num_cols, 6))
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # YOLO
        yolo_viz = bbox_viz.visualize_detections(image_path, yolo_res['detections'])
        if yolo_viz is not None:
            yolo_viz_rgb = cv2.cvtColor(yolo_viz, cv2.COLOR_BGR2RGB)
            axes[1].imshow(yolo_viz_rgb)
        axes[1].set_title(f'YOLO\n{yolo_res["num_detections"]} detections', 
                         fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # RT-DETR
        rtdetr_viz = bbox_viz.visualize_detections(image_path, rtdetr_res['detections'])
        if rtdetr_viz is not None:
            rtdetr_viz_rgb = cv2.cvtColor(rtdetr_viz, cv2.COLOR_BGR2RGB)
            axes[2].imshow(rtdetr_viz_rgb)
        axes[2].set_title(f'RT-DETR\n{rtdetr_res["num_detections"]} detections', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        # SAM (if available)
        if has_sam:
            # Find corresponding SAM result
            sam_res = None
            for s in sam_data['segmentations']:
                if s['image_name'] == image_name:
                    sam_res = s
                    break
            
            if sam_res:
                # Note: Can't visualize actual masks here since they're not in JSON
                # Just show detection count
                axes[3].imshow(image_rgb)
                axes[3].set_title(f'SAM Segmentation\n{sam_res["num_masks"]} masks', 
                                fontsize=12, fontweight='bold')
                axes[3].text(0.5, 0.5, 'See segmentation\nvisualizations folder', 
                           ha='center', va='center', transform=axes[3].transAxes,
                           fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                axes[3].imshow(image_rgb)
                axes[3].set_title('SAM\nNo data', fontsize=12, fontweight='bold')
            axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = output_dir / f'comparison_{image_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {output_path}")
    
    print(f"\nComparisons saved to: {output_dir}")

def main():
    print("=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    
    create_sample_comparisons(num_samples=5)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()