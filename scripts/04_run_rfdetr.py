#!/usr/bin/env python3
"""
Script 04: Run RT-DETR Inference
- Load preprocessed test images
- Run RT-DETR detection
- Save detection results and visualizations
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.models.rfdetr_model import RFDETRModel

def main():
    print("=" * 80)
    print("RT-DETR INFERENCE")
    print("=" * 80)
    
    # Get configuration
    model_name = config.get('models.rfdetr.model_name', 'rtdetr-l.pt')
    conf_threshold = config.get('models.rfdetr.conf_threshold', 0.25)
    iou_threshold = config.get('models.rfdetr.iou_threshold', 0.45)
    
    print(f"\nModel Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  IOU threshold: {iou_threshold}")
    
    # Initialize RT-DETR model
    print("\nInitializing RT-DETR model...")
    rtdetr = RFDETRModel(model_name, conf_threshold, iou_threshold)
    
    # Get test images path
    test_images = config.dataset_processed_path / 'test' / 'images'
    
    if not test_images.exists():
        print(f"\nError: Test images not found at {test_images}")
        print("Please run script 02_preprocess_data.py first")
        return
    
    # Count test images
    image_files = list(test_images.glob('*.jpg')) + list(test_images.glob('*.png'))
    print(f"\nFound {len(image_files)} test images")
    
    # Create output directory
    output_dir = config.detections_path / 'rtdetr'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    print("\nRunning RT-DETR inference...")
    start_time = time.time()
    
    results = rtdetr.predict_batch(
        str(test_images),
        str(output_dir),
        save_visualizations=True
    )
    
    inference_time = time.time() - start_time
    
    # Calculate statistics
    total_detections = sum(r['num_detections'] for r in results)
    avg_detections = total_detections / len(results) if results else 0
    avg_time_per_image = inference_time / len(results) if results else 0
    
    print(f"\n{'='*80}")
    print("RT-DETR INFERENCE RESULTS")
    print(f"{'='*80}")
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {avg_detections:.2f}")
    print(f"Total inference time: {inference_time:.2f}s")
    print(f"Average time per image: {avg_time_per_image:.3f}s")
    print(f"FPS: {1/avg_time_per_image:.2f}" if avg_time_per_image > 0 else "FPS: N/A")
    
    # Save results to JSON
    results_file = config.metrics_path / 'rtdetr_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'model_info': rtdetr.get_model_info(),
            'statistics': {
                'total_images': len(results),
                'total_detections': total_detections,
                'avg_detections_per_image': avg_detections,
                'total_inference_time': inference_time,
                'avg_time_per_image': avg_time_per_image,
                'fps': 1/avg_time_per_image if avg_time_per_image > 0 else 0
            },
            'detections': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("Next step: Run script 05_run_sam.py for SAM segmentation")
    print("=" * 80)

if __name__ == "__main__":
    main()