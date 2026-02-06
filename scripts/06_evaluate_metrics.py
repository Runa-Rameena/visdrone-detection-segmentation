#!/usr/bin/env python3
"""
Script 06: Evaluate Metrics and Generate Report
- Load ground truth annotations from test-dev
- Evaluate YOLO and RT-DETR detections
- Perform small object analysis
- Generate comparison visualizations
- Create final report

CRITICAL NOTES:
- YOLO/RT-DETR trained on COCO classes, VisDrone has different taxonomy
- Evaluator can optionally ignore_class_mismatch to focus on spatial accuracy
- Ground truth uses original image coordinates (no padding/resizing applied)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.dataset.parser import VisDroneParser
from src.evaluation.metrics import DetectionMetrics
from src.evaluation.small_object_analysis import SmallObjectAnalysis

def load_ground_truth_from_split(test_split_file: Path, annotations_dir: Path) -> dict:
    """
    Load ground truth annotations for test set by image ID
    
    Args:
        test_split_file: Path to test.txt containing image IDs
        annotations_dir: Path to annotations directory
        
    Returns:
        Dict mapping image_id -> list of ground truth annotations
    """
    print("Loading ground truth annotations...")
    
    # Read test split
    if not test_split_file.exists():
        print(f"Warning: Test split file not found: {test_split_file}")
        return {}
    
    with open(test_split_file, 'r') as f:
        test_images = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(test_images)} image IDs in test split")
    
    # Parse annotations using VisDroneParser
    parser = VisDroneParser(str(annotations_dir))
    
    ground_truths_by_id = {}
    not_found = 0
    
    for image_id in test_images:
        ann_file = annotations_dir / f"{image_id}.txt"
        if ann_file.exists():
            annotations = parser.parse_annotation(str(ann_file))
            if annotations:  # Only include if has annotations
                ground_truths_by_id[image_id] = annotations
        else:
            not_found += 1
    
    print(f"  Loaded ground truth for {len(ground_truths_by_id)} images")
    if not_found > 0:
        print(f"  Warning: {not_found} annotation files not found")
    
    return ground_truths_by_id

def load_predictions_from_json(results_file: Path) -> dict:
    """
    Load predictions from JSON file, indexed by image_id
    
    Args:
        results_file: Path to JSON results file (e.g., yolo_results.json)
        
    Returns:
        Dict mapping image_id -> list of predictions
    """
    print(f"Loading predictions from {results_file.name}...")
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return {}
    
    with open(results_file, 'r') as f:
        model_data = json.load(f)
    
    predictions_by_id = {}
    
    # Handle different JSON formats
    if 'detections' in model_data:
        # Format: {"detections": [{"image_id": ..., "detections": [...]}, ...]}
        for det_result in model_data['detections']:
            # Extract image_id from image_path or use direct image_id
            image_path = det_result.get('image_path', '')
            image_id = det_result.get('image_id')
            
            if not image_id and image_path:
                # Try to extract from path: ".../images/0000006_00159_d_0000001.jpg" -> "0000006_00159_d_0000001"
                image_id = Path(image_path).stem
            
            detections = det_result.get('detections', [])
            if image_id and detections:
                predictions_by_id[image_id] = detections
    
    print(f"  Loaded predictions for {len(predictions_by_id)} images")
    return predictions_by_id

def evaluate_model(model_name: str, results_file: Path, ground_truths_by_id: dict, config):
    """Evaluate a detection model"""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}")
    print(f"{'='*80}")
    
    # Load model results
    predictions_by_id = load_predictions_from_json(results_file)
    
    if not predictions_by_id:
        print(f"Error: No predictions loaded for {model_name}")
        return None
    
    if not ground_truths_by_id:
        print(f"Error: No ground truth annotations")
        return None
    
    # Initialize evaluators
    iou_threshold = config.get('evaluation.iou_threshold', 0.5)
    small_threshold = config.get('evaluation.small_object_threshold', 32)
    
    metrics_calc = DetectionMetrics(iou_threshold=iou_threshold)
    small_obj_analyzer = SmallObjectAnalysis(small_threshold=small_threshold)
    
    print(f"\nMetrics configuration:")
    print(f"  IoU threshold: {iou_threshold}")
    print(f"  Small object threshold: {small_threshold}x{small_threshold}")
    
    # Evaluate detection metrics
    print("\nCalculating detection metrics...")
    print(f"  Note: Using ignore_class_mismatch=True")
    print(f"  Reason: YOLO trained on COCO, GT is VisDrone (different taxonomies)")
    
    metrics = metrics_calc.evaluate_dataset(
        predictions_by_id, 
        ground_truths_by_id,
        ignore_class_mismatch=True  # Important: YOLO trained on COCO, GT is VisDrone
    )
    
    print("\nOverall Metrics:")
    print(f"  Precision: {metrics['overall']['precision']:.4f}")
    print(f"  Recall: {metrics['overall']['recall']:.4f}")
    print(f"  F1 Score: {metrics['overall']['f1_score']:.4f}")
    print(f"  mAP: {metrics['overall']['mAP']:.4f}")
    print(f"  True Positives: {metrics['overall']['total_tp']}")
    print(f"  False Positives: {metrics['overall']['total_fp']}")
    print(f"  False Negatives: {metrics['overall']['total_fn']}")
    
    # Small object analysis
    print("\nPerforming small object analysis...")
    small_obj_analysis = small_obj_analyzer.analyze_dataset(predictions_by_id, ground_truths_by_id)
    
    print("\nSmall Object Analysis (Predictions):")
    for category, stats in small_obj_analysis.get('predictions', {}).items():
        print(f"  {category.capitalize()}:")
        print(f"    Total detections: {stats['total_detections']}")
        print(f"    Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"    Mean area: {stats['mean_area']:.2f} pixels")
    
    if 'detection_rates' in small_obj_analysis:
        print("\nDetection Rates by Object Size:")
        for category, rate_data in small_obj_analysis['detection_rates'].items():
            print(f"  {category.capitalize()}:")
            print(f"    Ground truth: {rate_data['ground_truth_count']}")
            print(f"    Detected: {rate_data['detection_count']}")
            if rate_data['ground_truth_count'] > 0:
                print(f"    Detection rate: {rate_data['detection_rate']:.4f} ({rate_data['detection_rate']*100:.2f}%)")
    
    # Read inference statistics if available
    with open(results_file, 'r') as f:
        model_data = json.load(f)
    
    inference_stats = model_data.get('statistics', {})
    
    if inference_stats:
        print("\nInference Statistics:")
        if 'total_images' in inference_stats:
            print(f"  Total images: {inference_stats['total_images']}")
        if 'total_detections' in inference_stats:
            print(f"  Total detections: {inference_stats['total_detections']}")
        if 'avg_detections_per_image' in inference_stats:
            print(f"  Avg detections/image: {inference_stats['avg_detections_per_image']:.2f}")
        if 'avg_time_per_image' in inference_stats:
            print(f"  Average time/image: {inference_stats['avg_time_per_image']:.3f}s")
        if 'fps' in inference_stats and inference_stats['fps'] > 0:
            print(f"  FPS: {inference_stats['fps']:.2f}")
    
    # Save evaluation results
    eval_results = {
        'model': model_name,
        'metrics': metrics,
        'small_object_analysis': small_obj_analysis,
        'inference_stats': inference_stats,
        'num_images': len(ground_truths_by_id),
        'num_predictions': len(predictions_by_id)
    }
    
    output_file = config.metrics_path / f'{model_name.lower()}_evaluation.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")
    
    return eval_results

def compare_models(yolo_eval: dict, rtdetr_eval: dict, config):
    """Generate comparison between models"""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")
    
    if not yolo_eval or not rtdetr_eval:
        print("Error: Could not compare models (missing evaluation data)")
        return
    
    print("\nOverall Performance Comparison:")
    print(f"{'Metric':<20} {'YOLO':<15} {'RT-DETR':<15}")
    print("-" * 50)
    
    yolo_metrics = yolo_eval['metrics']['overall']
    rtdetr_metrics = rtdetr_eval['metrics']['overall']
    
    for metric in ['precision', 'recall', 'f1_score', 'mAP']:
        yolo_val = yolo_metrics[metric]
        rtdetr_val = rtdetr_metrics[metric]
        print(f"{metric:<20} {yolo_val:<15.4f} {rtdetr_val:<15.4f}")
    
    print("\nInference Speed Comparison:")
    yolo_stats = yolo_eval['inference_stats']
    rtdetr_stats = rtdetr_eval['inference_stats']
    
    print(f"{'Metric':<25} {'YOLO':<15} {'RT-DETR':<15}")
    print("-" * 55)
    
    if 'avg_time_per_image' in yolo_stats and 'avg_time_per_image' in rtdetr_stats:
        print(f"{'Avg time per image (s)':<25} {yolo_stats['avg_time_per_image']:<15.3f} {rtdetr_stats['avg_time_per_image']:<15.3f}")
    
    if 'fps' in yolo_stats and 'fps' in rtdetr_stats:
        print(f"{'FPS':<25} {yolo_stats['fps']:<15.2f} {rtdetr_stats['fps']:<15.2f}")
    
    print("\nSmall Object Detection Comparison:")
    yolo_small = yolo_eval['small_object_analysis']['predictions'].get('small', {})
    rtdetr_small = rtdetr_eval['small_object_analysis']['predictions'].get('small', {})
    
    if yolo_small and rtdetr_small:
        print(f"{'Metric':<25} {'YOLO':<15} {'RT-DETR':<15}")
        print("-" * 55)
        print(f"{'Small obj detections':<25} {yolo_small['total_detections']:<15} {rtdetr_small['total_detections']:<15}")
        print(f"{'Mean confidence':<25} {yolo_small['mean_confidence']:<15.4f} {rtdetr_small['mean_confidence']:<15.4f}")
        
        if 'mean_area' in yolo_small and 'mean_area' in rtdetr_small:
            print(f"{'Mean area (pixels)':<25} {yolo_small['mean_area']:<15.2f} {rtdetr_small['mean_area']:<15.2f}")
    
    # Save comparison
    comparison = {
        'overall_metrics': {
            'yolo': yolo_metrics,
            'rtdetr': rtdetr_metrics
        },
        'inference_stats': {
            'yolo': yolo_stats,
            'rtdetr': rtdetr_stats
        },
        'small_object_analysis': {
            'yolo': yolo_eval['small_object_analysis'],
            'rtdetr': rtdetr_eval['small_object_analysis']
        }
    }
    
    output_file = config.metrics_path / 'model_comparison.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {output_file}")

def main():
    print("=" * 80)
    print("EVALUATION AND METRICS CALCULATION")
    print("=" * 80)
    
    print("\nIMPORTANT NOTES:")
    print("  ✓ Matches predictions and GT by image_id (not array position)")
    print("  ✓ Uses ignore_class_mismatch=True (YOLO vs VisDrone taxonomy)")
    print("  ✓ Focuses on spatial accuracy (IoU)")
    print("  ✓ Performs small-object analysis for VisDrone challenges")
    
    # Define paths
    config.metrics_path.mkdir(parents=True, exist_ok=True)
    
    # Check if detection results exist
    yolo_results = config.metrics_path / 'yolo_results.json'
    rtdetr_results = config.metrics_path / 'rtdetr_results.json'
    
    yolo_exists = yolo_results.exists()
    rtdetr_exists = rtdetr_results.exists()
    
    print(f"\n{'✓' if yolo_exists else '✗'} YOLO results: {yolo_results.name}")
    print(f"{'✓' if rtdetr_exists else '✗'} RT-DETR results: {rtdetr_results.name}")
    
    if not (yolo_exists or rtdetr_exists):
        print("\nError: No detection results found")
        print("Please run:")
        print("  - script 03_run_yolo.py for YOLO inference")
        print("  - script 04_run_rfdetr.py for RT-DETR inference")
        return
    
    # Load ground truth from test-dev
    test_split_file = config.splits_path / 'test.txt'
    annotations_dir = config.raw_path / 'VisDrone2019-DET-test-dev' / 'annotations'
    
    if not annotations_dir.exists():
        print(f"\nError: Annotations directory not found at {annotations_dir}")
        print("Expected structure: data/raw/VisDrone2019-DET-test-dev/annotations/")
        return
    
    ground_truths_by_id = load_ground_truth_from_split(test_split_file, annotations_dir)
    
    if not ground_truths_by_id:
        print("Error: Could not load ground truth annotations")
        return
    
    yolo_eval = None
    rtdetr_eval = None
    
    # Evaluate YOLO
    if yolo_exists:
        yolo_eval = evaluate_model('YOLO', yolo_results, ground_truths_by_id, config)
    
    # Evaluate RT-DETR
    if rtdetr_exists:
        rtdetr_eval = evaluate_model('RTDETR', rtdetr_results, ground_truths_by_id, config)
    
    # Compare models if both available
    if yolo_eval and rtdetr_eval:
        compare_models(yolo_eval, rtdetr_eval, config)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {config.metrics_path}")

if __name__ == "__main__":
    main()
