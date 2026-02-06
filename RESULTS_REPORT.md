# VisDrone Detection & Segmentation Project Report

## Executive Summary

This report presents the results of object detection and segmentation experiments on the VisDrone2019-DET dataset using YOLO, RT-DETR, and SAM models.

## Dataset Information

- **Dataset**: VisDrone2019-DET
- **Image Resolution**: 640x640 (preprocessed)
- **Split Ratio**: 70% train, 20% validation, 10% test
- **Test Set Size**: 1610 images

## Model Performance Comparison

### Overall Detection Metrics

| Metric | YOLO | RT-DETR |
|--------|------|---------|
| Precision | 0.0026 | 0.0011 |
| Recall | 0.0003 | 0.0006 |
| F1 Score | 0.0005 | 0.0008 |
| mAP | 0.0037 | 0.0059 |

### Inference Speed

| Metric | YOLO | RT-DETR |
|--------|------|---------|
| Average Time per Image (s) | 0.344 | 2.255 |
| FPS | 2.90 | 0.44 |
| Total Inference Time (s) | 554.47 | 3630.00 |

### Detection Counts

| Metric | YOLO | RT-DETR |
|--------|------|---------|
| Total Detections | 7360 | 43428 |
| Avg Detections per Image | 4.57 | 26.97 |

## Small Object Analysis

### YOLO Performance by Object Size

#### Medium Objects
- Total Detections: 2286
- Mean Confidence: 0.5028
- Mean Area: 2521.19 pixels²

#### Small Objects
- Total Detections: 4605
- Mean Confidence: 0.4345
- Mean Area: 479.60 pixels²

#### Large Objects
- Total Detections: 469
- Mean Confidence: 0.4447
- Mean Area: 98095.66 pixels²

### RT-DETR Performance by Object Size

#### Medium Objects
- Total Detections: 7085
- Mean Confidence: 0.5262
- Mean Area: 2381.01 pixels²

#### Small Objects
- Total Detections: 35924
- Mean Confidence: 0.4071
- Mean Area: 310.42 pixels²

#### Large Objects
- Total Detections: 419
- Mean Confidence: 0.4744
- Mean Area: 28797.76 pixels²

## SAM Segmentation

SAM segmentation results not available.
