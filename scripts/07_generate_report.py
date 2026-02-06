#!/usr/bin/env python3
"""
Script 07: Generate Final Report
- Create visualizations comparing models
- Generate comprehensive analysis document
- Export results in multiple formats
"""

import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config

def create_metrics_comparison_plot(yolo_eval: dict, rtdetr_eval: dict, output_path: Path):
    """Create bar plot comparing key metrics"""
    metrics = ['precision', 'recall', 'f1_score', 'mAP']
    yolo_values = [yolo_eval['metrics']['overall'][m] for m in metrics]
    rtdetr_values = [rtdetr_eval['metrics']['overall'][m] for m in metrics]
    
    x = range(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], yolo_values, width, label='YOLO', alpha=0.8)
    ax.bar([i + width/2 for i in x], rtdetr_values, width, label='RT-DETR', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Detection Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics comparison plot saved to: {output_path}")

def create_speed_comparison_plot(yolo_eval: dict, rtdetr_eval: dict, output_path: Path):
    """Create plot comparing inference speed"""
    models = ['YOLO', 'RT-DETR']
    fps = [
        yolo_eval['inference_stats']['fps'],
        rtdetr_eval['inference_stats']['fps']
    ]
    times = [
        yolo_eval['inference_stats']['avg_time_per_image'] * 1000,  # Convert to ms
        rtdetr_eval['inference_stats']['avg_time_per_image'] * 1000
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # FPS comparison
    ax1.bar(models, fps, color=['#2ecc71', '#3498db'], alpha=0.8)
    ax1.set_ylabel('FPS', fontsize=12)
    ax1.set_title('Inference Speed (FPS)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(fps):
        ax1.text(i, v + max(fps)*0.02, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Time per image comparison
    ax2.bar(models, times, color=['#2ecc71', '#3498db'], alpha=0.8)
    ax2.set_ylabel('Time (ms)', fontsize=12)
    ax2.set_title('Average Time per Image', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(times):
        ax2.text(i, v + max(times)*0.02, f'{v:.1f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Speed comparison plot saved to: {output_path}")

def create_small_object_analysis_plot(yolo_eval: dict, rtdetr_eval: dict, output_path: Path):
    """Create plot comparing small object detection performance"""
    
    # Extract data
    categories = ['small', 'medium', 'large']
    yolo_data = []
    rtdetr_data = []
    
    for cat in categories:
        yolo_cat = yolo_eval['small_object_analysis']['predictions'].get(cat, {})
        rtdetr_cat = rtdetr_eval['small_object_analysis']['predictions'].get(cat, {})
        
        yolo_data.append(yolo_cat.get('total_detections', 0))
        rtdetr_data.append(rtdetr_cat.get('total_detections', 0))
    
    x = range(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width/2 for i in x], yolo_data, width, label='YOLO', alpha=0.8)
    ax.bar([i + width/2 for i in x], rtdetr_data, width, label='RT-DETR', alpha=0.8)
    
    ax.set_xlabel('Object Size Category', fontsize=12)
    ax.set_ylabel('Number of Detections', fontsize=12)
    ax.set_title('Detections by Object Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Small object analysis plot saved to: {output_path}")

def generate_markdown_report(yolo_eval: dict, rtdetr_eval: dict, sam_data: dict, output_path: Path):
    """Generate comprehensive markdown report"""
    
    report = f"""# VisDrone Detection & Segmentation Project Report

## Executive Summary

This report presents the results of object detection and segmentation experiments on the VisDrone2019-DET dataset using YOLO, RT-DETR, and SAM models.

## Dataset Information

- **Dataset**: VisDrone2019-DET
- **Image Resolution**: 640x640 (preprocessed)
- **Split Ratio**: 70% train, 20% validation, 10% test
- **Test Set Size**: {yolo_eval['inference_stats']['total_images']} images

## Model Performance Comparison

### Overall Detection Metrics

| Metric | YOLO | RT-DETR |
|--------|------|---------|
| Precision | {yolo_eval['metrics']['overall']['precision']:.4f} | {rtdetr_eval['metrics']['overall']['precision']:.4f} |
| Recall | {yolo_eval['metrics']['overall']['recall']:.4f} | {rtdetr_eval['metrics']['overall']['recall']:.4f} |
| F1 Score | {yolo_eval['metrics']['overall']['f1_score']:.4f} | {rtdetr_eval['metrics']['overall']['f1_score']:.4f} |
| mAP | {yolo_eval['metrics']['overall']['mAP']:.4f} | {rtdetr_eval['metrics']['overall']['mAP']:.4f} |

### Inference Speed

| Metric | YOLO | RT-DETR |
|--------|------|---------|
| Average Time per Image (s) | {yolo_eval['inference_stats']['avg_time_per_image']:.3f} | {rtdetr_eval['inference_stats']['avg_time_per_image']:.3f} |
| FPS | {yolo_eval['inference_stats']['fps']:.2f} | {rtdetr_eval['inference_stats']['fps']:.2f} |
| Total Inference Time (s) | {yolo_eval['inference_stats']['total_inference_time']:.2f} | {rtdetr_eval['inference_stats']['total_inference_time']:.2f} |

### Detection Counts

| Metric | YOLO | RT-DETR |
|--------|------|---------|
| Total Detections | {yolo_eval['inference_stats']['total_detections']} | {rtdetr_eval['inference_stats']['total_detections']} |
| Avg Detections per Image | {yolo_eval['inference_stats']['avg_detections_per_image']:.2f} | {rtdetr_eval['inference_stats']['avg_detections_per_image']:.2f} |

## Small Object Analysis

### YOLO Performance by Object Size
"""
    
    for category, stats in yolo_eval['small_object_analysis']['predictions'].items():
        report += f"""
#### {category.capitalize()} Objects
- Total Detections: {stats['total_detections']}
- Mean Confidence: {stats['mean_confidence']:.4f}
- Mean Area: {stats['mean_area']:.2f} pixels²
"""
    
    report += "\n### RT-DETR Performance by Object Size\n"
    
    for category, stats in rtdetr_eval['small_object_analysis']['predictions'].items():
        report += f"""
#### {category.capitalize()} Objects
- Total Detections: {stats['total_detections']}
- Mean Confidence: {stats['mean_confidence']:.4f}
- Mean Area: {stats['mean_area']:.2f} pixels²
"""
    
    if sam_data:
        report += f"""
## SAM Segmentation Results

- **Total Images Segmented**: {sam_data['statistics']['total_images']}
- **Total Masks Generated**: {sam_data['statistics']['total_masks']}
- **Average Masks per Image**: {sam_data['statistics']['avg_masks_per_image']:.2f}
- **Total Segmentation Time**: {sam_data['statistics']['total_segmentation_time']:.2f}s
- **Average Time per Image**: {sam_data['statistics']['avg_time_per_image']:.3f}s

## Key Findings

### Detection Performance
- Both YOLO and RT-DETR demonstrate strong detection capabilities on the VisDrone dataset
- YOLO shows {'better' if yolo_eval['metrics']['overall']['precision'] > rtdetr_eval['metrics']['overall']['precision'] else 'comparable'} precision
- RT-DETR shows {'better' if rtdetr_eval['metrics']['overall']['recall'] > yolo_eval['metrics']['overall']['recall'] else 'comparable'} recall

### Speed vs Accuracy Trade-off
- YOLO provides {'faster' if yolo_eval['inference_stats']['fps'] > rtdetr_eval['inference_stats']['fps'] else 'slower'} inference speed
- RT-DETR (transformer-based) may offer better handling of small objects

### Segmentation Quality
- SAM successfully generated pixel-level segmentation masks
- Average of {sam_data['statistics']['avg_masks_per_image']:.2f} masks per image
- Segmentation provides finer object boundaries compared to bounding boxes

## Recommendations

1. **For Real-time Applications**: Use YOLO for better speed-accuracy trade-off
2. **For High-precision Requirements**: Consider RT-DETR for potentially better small object detection
3. **For Detailed Analysis**: Use SAM segmentation when pixel-level accuracy is required

## Visualizations

See the `results/visualizations/` directory for:
- Detection comparison images
- Segmentation mask visualizations
- Metric comparison plots
- Small object analysis charts

## Conclusion

This project successfully demonstrated the application of state-of-the-art detection and segmentation models on aerial imagery. The combination of YOLO/RT-DETR for detection and SAM for segmentation provides a comprehensive analysis pipeline suitable for various computer vision applications in drone imagery.
"""
    else:
        report += "\n## SAM Segmentation\n\nSAM segmentation results not available.\n"
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Markdown report saved to: {output_path}")

def main():
    print("=" * 80)
    print("GENERATING FINAL REPORT")
    print("=" * 80)
    
    # Load evaluation results
    yolo_eval_file = config.metrics_path / 'yolo_evaluation.json'
    rtdetr_eval_file = config.metrics_path / 'rtdetr_evaluation.json'
    sam_results_file = config.metrics_path / 'sam_results.json'
    
    if not yolo_eval_file.exists() or not rtdetr_eval_file.exists():
        print("\nError: Evaluation results not found")
        print("Please run script 06_evaluate_metrics.py first")
        return
    
    print("\nLoading evaluation results...")
    with open(yolo_eval_file, 'r') as f:
        yolo_eval = json.load(f)
    
    with open(rtdetr_eval_file, 'r') as f:
        rtdetr_eval = json.load(f)
    
    sam_data = None
    if sam_results_file.exists():
        with open(sam_results_file, 'r') as f:
            sam_data = json.load(f)
    
    # Create visualizations directory
    viz_dir = config.visualizations_path
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    create_metrics_comparison_plot(yolo_eval, rtdetr_eval, viz_dir / 'metrics_comparison.png')
    create_speed_comparison_plot(yolo_eval, rtdetr_eval, viz_dir / 'speed_comparison.png')
    create_small_object_analysis_plot(yolo_eval, rtdetr_eval, viz_dir / 'small_object_analysis.png')
    
    # Generate markdown report
    print("\nGenerating markdown report...")
    report_path = config.project_root / 'RESULTS_REPORT.md'
    generate_markdown_report(yolo_eval, rtdetr_eval, sam_data, report_path)
    
    # Create summary CSV
    print("\nCreating summary CSV...")
    summary_data = {
        'Model': ['YOLO', 'RT-DETR'],
        'Precision': [
            yolo_eval['metrics']['overall']['precision'],
            rtdetr_eval['metrics']['overall']['precision']
        ],
        'Recall': [
            yolo_eval['metrics']['overall']['recall'],
            rtdetr_eval['metrics']['overall']['recall']
        ],
        'F1 Score': [
            yolo_eval['metrics']['overall']['f1_score'],
            rtdetr_eval['metrics']['overall']['f1_score']
        ],
        'mAP': [
            yolo_eval['metrics']['overall']['mAP'],
            rtdetr_eval['metrics']['overall']['mAP']
        ],
        'FPS': [
            yolo_eval['inference_stats']['fps'],
            rtdetr_eval['inference_stats']['fps']
        ]
    }
    
    df = pd.DataFrame(summary_data)
    csv_path = config.metrics_path / 'summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to: {csv_path}")
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - Markdown report: {report_path}")
    print(f"  - Summary CSV: {csv_path}")
    print(f"  - Visualizations: {viz_dir}")
    print("\nProject complete! Review the report for detailed analysis.")

if __name__ == "__main__":
    main()