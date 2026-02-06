#!/usr/bin/env python3
"""
Script 05: Run SAM Segmentation
- Load YOLO detection results as prompts
- Run SAM segmentation
- Save mask results and visualizations

NOTE: SAM is OPTIONAL for the pipeline.
- Used for QUALITATIVE segmentation visualization only
- NOT used for any metric evaluation
- If SAM fails to load, pipeline can continue
"""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.models.sam_model import SAMModel
from src.utils.checkpoint_manager import CheckpointManager

def main():
    print("=" * 80)
    print("SAM SEGMENTATION (OPTIONAL QUALITATIVE STEP)")
    print("=" * 80)
    
    # Get configuration
    model_type = config.get('models.sam.model_type', 'vit_h')
    checkpoint = config.get('models.sam.checkpoint', 'sam_vit_h_4b8939.pth')
    
    print(f"\nModel Configuration:")
    print(f"  Model type: {model_type}")
    print(f"  Checkpoint: {checkpoint}")
    
    checkpoint_path = Path(checkpoint)
    
    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"\n✗ SAM checkpoint not found at {checkpoint}")
        print("\n📥 DOWNLOAD INSTRUCTIONS:")
        print("\n  Option 1 - Automated Download (Recommended):")
        print(f"    python -m src.utils.checkpoint_manager --checkpoint {checkpoint}")
        print("\n  Option 2 - Manual Download:")
        checkpoint_info = CheckpointManager.get_checkpoint_info(checkpoint)
        if checkpoint_info:
            print(f"    Download: {checkpoint_info['url']}")
            print(f"    Save as:  {checkpoint}")
            print(f"    Size:     {checkpoint_info['size'] / (1024**3):.2f} GB")
        print("\n⚠️  SAM segmentation SKIPPED (optional step)")
        print("   Pipeline can continue without SAM results")
        return True
    
    # Validate checkpoint before loading
    print("\n" + "=" * 80)
    print("VALIDATING CHECKPOINT")
    print("=" * 80)
    
    checkpoint_name = checkpoint_path.name
    checkpoint_info = CheckpointManager.get_checkpoint_info(checkpoint_name)
    
    # Check file existence
    print(f"✓ File exists: {checkpoint}")
    
    # Check file size
    if checkpoint_info:
        size_valid, size_msg = CheckpointManager.validate_checkpoint_size(
            checkpoint_path, 
            checkpoint_info['size']
        )
        print(size_msg)
        
        if not size_valid:
            print(f"\n✗ CHECKPOINT VALIDATION FAILED")
            print(f"\nThe SAM checkpoint appears to be corrupted or incomplete.")
            print(f"\n📥 REPAIR INSTRUCTIONS:")
            print(f"\n  Step 1: Delete the corrupted file:")
            print(f"    del {checkpoint}")
            print(f"\n  Step 2: Re-download:")
            print(f"    python -m src.utils.checkpoint_manager --checkpoint {checkpoint}")
            print(f"\n  Step 3: Re-run this script:")
            print(f"    python scripts/05_run_sam.py")
            print(f"\n⚠️  SAM segmentation SKIPPED (optional step)")
            print(f"   Pipeline can continue without SAM results")
            return True
    
    # Check PyTorch loadability (most important check)
    print("Testing PyTorch checkpoint loading...")
    pytorch_valid, pytorch_msg = CheckpointManager.can_load_pytorch_checkpoint(checkpoint_path)
    print(pytorch_msg)
    
    if not pytorch_valid:
        print(f"\n✗ CHECKPOINT VALIDATION FAILED")
        print(f"\nPyTorch cannot load the checkpoint - file is corrupted.")
        print(f"\n📥 REPAIR INSTRUCTIONS:")
        print(f"\n  Step 1: Delete the corrupted file:")
        print(f"    del {checkpoint}")
        print(f"\n  Step 2: Re-download:")
        print(f"    python -m src.utils.checkpoint_manager --checkpoint {checkpoint}")
        print(f"\n  Step 3: Re-run this script:")
        print(f"    python scripts/05_run_sam.py")
        print(f"\n⚠️  SAM segmentation SKIPPED (optional step)")
        print(f"   Pipeline can continue without SAM results")
        return True
    
    print("✓ Checkpoint validation passed (file size + PyTorch loadability OK)")
    
    # Try to initialize SAM model
    print("\n" + "=" * 80)
    print("INITIALIZING SAM MODEL")
    print("=" * 80)
    
    sam = SAMModel(model_type, checkpoint, validate_checkpoint=False)
    
    if not sam.predictor:
        print(f"\n✗ SAM MODEL LOADING FAILED")
        print(f"\nThe checkpoint exists but PyTorch cannot load it.")
        print(f"This indicates file corruption.\n")
        print(f"📥 SOLUTION:")
        print(f"  1. Delete: del {checkpoint}")
        print(f"  2. Re-download: python -m src.utils.checkpoint_manager --checkpoint {checkpoint}")
        print(f"  3. Retry: python scripts/05_run_sam.py")
        print(f"\n⚠️  SAM segmentation SKIPPED (optional step)")
        print(f"   Pipeline can continue without SAM results")
        return True
    
    # Load YOLO detection results (use as prompts)
    yolo_results_file = config.metrics_path / 'yolo_results.json'
    
    if not yolo_results_file.exists():
        print(f"\n✗ YOLO results not found at {yolo_results_file}")
        print("Please run script 03_run_yolo.py first")
        return False
    
    print(f"\nLoading YOLO detection results from {yolo_results_file}...")
    with open(yolo_results_file, 'r') as f:
        yolo_data = json.load(f)
    
    detections = yolo_data['detections']
    print(f"Loaded detections for {len(detections)} images")
    
    # Get test images path
    test_images = config.dataset_processed_path / 'test' / 'images'
    
    # Create output directory
    output_dir = config.segmentations_path / 'sam'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run SAM segmentation
    print("\n" + "=" * 80)
    print("RUNNING SAM SEGMENTATION")
    print("=" * 80)
    print("\nGenerating masks from YOLO detections...")
    print("(This is a qualitative step - results not used for metrics)\n")
    
    start_time = time.time()
    
    results = sam.predict_batch_with_detections(
        str(test_images),
        detections,
        str(output_dir),
        save_visualizations=True
    )
    
    segmentation_time = time.time() - start_time
    
    # Calculate statistics
    total_masks = sum(r['num_masks'] for r in results)
    avg_masks = total_masks / len(results) if results else 0
    avg_time_per_image = segmentation_time / len(results) if results else 0
    
    print(f"\n{'='*80}")
    print("SAM SEGMENTATION RESULTS")
    print(f"{'='*80}")
    print(f"Total images processed: {len(results)}")
    print(f"Total masks generated: {total_masks}")
    print(f"Average masks per image: {avg_masks:.2f}")
    print(f"Total segmentation time: {segmentation_time:.2f}s")
    print(f"Average time per image: {avg_time_per_image:.3f}s")
    
    # Save results to JSON
    results_file = config.metrics_path / 'sam_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON (convert numpy arrays to lists)
    json_results = []
    for r in results:
        json_r = {
            'image_path': r['image_path'],
            'image_name': r.get('image_name', ''),
            'num_masks': r['num_masks'],
            'masks': []
        }
        for mask_dict in r.get('masks', []):
            json_r['masks'].append({
                'bbox': mask_dict['bbox'],
                'score': mask_dict['score']
                # Note: mask array not saved to JSON (too large)
            })
        json_results.append(json_r)
    
    with open(results_file, 'w') as f:
        json.dump({
            'model_info': sam.get_model_info(),
            'statistics': {
                'total_images': len(results),
                'total_masks': total_masks,
                'avg_masks_per_image': avg_masks,
                'total_segmentation_time': segmentation_time,
                'avg_time_per_image': avg_time_per_image
            },
            'note': 'SAM segmentation is qualitative only - not used for metric evaluation',
            'segmentations': json_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Visualizations saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("✓ SAM segmentation complete")
    print("=" * 80)
    print("Next step: Run script 06_evaluate_metrics.py for evaluation")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Unexpected error in SAM segmentation:")
        print(f"  {type(e).__name__}: {str(e)}")
        print(f"\n⚠️  SAM segmentation SKIPPED")
        print(f"   Pipeline can continue without SAM results")
        sys.exit(0)  # Exit 0 to allow pipeline to continue
