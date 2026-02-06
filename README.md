# VisDrone Detection & Segmentation Project

Advanced computer vision project for object detection and segmentation on aerial/drone imagery using YOLO, RT-DETR, and SAM models.

## 🎯 Project Overview

This project implements a complete pipeline for:
- **Object Detection** using YOLO and RT-DETR (transformer-based)
- **Segmentation** using SAM (Segment Anything Model)
- **Evaluation** with comprehensive metrics (mAP, precision, recall, IoU)
- **Small Object Analysis** focusing on detection performance for small and dense objects
- **Comparative Analysis** between bounding box and mask-based approaches

## 📂 Project Structure

```
visdrone-detection-segmentation/
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/                  # Place your downloaded VisDrone dataset here
│   ├── processed/            # Preprocessed images and annotations
│   └── splits/               # Train/val/test split files
│
├── src/
│   ├── config.py
│   ├── dataset/
│   │   ├── parser.py         # VisDrone annotation parser
│   │   └── preprocessor.py   # Image preprocessing and splitting
│   ├── models/
│   │   ├── yolo_model.py     # YOLO wrapper
│   │   ├── rfdetr_model.py   # RT-DETR wrapper
│   │   └── sam_model.py      # SAM wrapper
│   ├── evaluation/
│   │   ├── metrics.py        # Detection metrics (mAP, precision, recall)
│   │   └── small_object_analysis.py
│   └── visualization/
│       ├── bbox_visualizer.py
│       └── mask_visualizer.py
│
├── scripts/
│   ├── 02_preprocess_data.py
│   ├── 03_run_yolo.py
│   ├── 04_run_rfdetr.py
│   ├── 05_run_sam.py
│   ├── 06_evaluate_metrics.py
│   └── 07_generate_report.py
│
└── results/
    ├── detections/           # Detection visualizations
    ├── segmentations/        # Segmentation visualizations
    ├── metrics/              # Evaluation results (JSON, CSV)
    └── visualizations/       # Comparison plots
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download SAM checkpoint (for segmentation)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 2. Dataset Setup

Download the VisDrone2019-DET dataset and place it in `data/raw/`:

```
data/raw/
├── images/           # Put all .jpg images here
└── annotations/      # Put all .txt annotation files here
```

### 3. Run the Pipeline

Execute scripts in order:

```bash
# Step 1: Preprocess dataset (resize, split, augment)
python scripts/02_preprocess_data.py

# Step 2: Run YOLO detection
python scripts/03_run_yolo.py

# Step 3: Run RT-DETR detection
python scripts/04_run_rfdetr.py

# Step 4: Run SAM segmentation (using YOLO detections as prompts)
python scripts/05_run_sam.py

# Step 5: Evaluate metrics and compare models
python scripts/06_evaluate_metrics.py

# Step 6: Generate final report with visualizations
python scripts/07_generate_report.py
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- Image preprocessing parameters (target size, augmentation)
- Model selection and confidence thresholds
- Evaluation metrics and IoU thresholds
- Output paths

## 📊 Key Features

### Object Detection
- **YOLO**: Fast, efficient detection for real-time applications
- **RT-DETR**: Transformer-based detection with improved small object handling

### Segmentation
- **SAM**: Pixel-level segmentation using detection bounding boxes as prompts
- High-quality masks for precise object boundaries

### Evaluation
- Standard metrics: Precision, Recall, mAP, F1 Score
- IoU-based matching between predictions and ground truth
- Per-class performance analysis

### Small Object Analysis
- Categorization by object size (small, medium, large)
- Detection rate analysis for each category
- Confidence distribution by object size

### Comparative Analysis
- Side-by-side model comparisons
- Speed vs accuracy trade-offs
- Bounding box vs segmentation mask quality

## 📈 Output Files

After running the complete pipeline, you'll find:

### Detection Results
- `results/detections/yolo/` - YOLO detection visualizations
- `results/detections/rtdetr/` - RT-DETR detection visualizations

### Segmentation Results
- `results/segmentations/sam/` - SAM segmentation masks

### Metrics
- `results/metrics/yolo_evaluation.json` - Complete YOLO evaluation
- `results/metrics/rtdetr_evaluation.json` - Complete RT-DETR evaluation
- `results/metrics/model_comparison.json` - Side-by-side comparison
- `results/metrics/summary.csv` - Summary table

### Visualizations
- `results/visualizations/metrics_comparison.png` - Performance metrics plot
- `results/visualizations/speed_comparison.png` - Inference speed comparison
- `results/visualizations/small_object_analysis.png` - Object size analysis

### Report
- `RESULTS_REPORT.md` - Comprehensive markdown report with all findings

## 🔧 Customization

### Using Different Models

Edit `config.yaml`:
```yaml
models:
  yolo:
    model_name: "yolov8n.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  rfdetr:
    model_name: "rtdetr-l.pt"  # Options: rtdetr-l, rtdetr-x
```

### Adjusting Preprocessing

```yaml
preprocessing:
  target_size: [640, 640]
  split_ratios:
    train: 0.7
    val: 0.2
    test: 0.1
```

### Custom Small Object Threshold

```yaml
evaluation:
  small_object_threshold: 32  # Objects smaller than 32x32 pixels
```

## 📝 VisDrone Annotation Format

VisDrone uses the following annotation format (CSV):
```
<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
```

Classes:
- 0: ignored regions
- 1: pedestrian
- 2: people
- 3: bicycle
- 4: car
- 5: van
- 6: truck
- 7: tricycle
- 8: awning-tricycle
- 9: bus
- 10: motor

## 🎓 For Hackathons/Labs

This project is structured for easy demonstration:

1. **Quick Results**: All scripts run sequentially with progress bars
2. **Visual Outputs**: Automatic generation of comparison visualizations
3. **Comprehensive Report**: Markdown report with all metrics and insights
4. **Reproducible**: Fixed random seeds and documented configuration

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in model inference
- Use smaller models (yolov8n instead of yolov8x)

### SAM Checkpoint Not Found
```bash
# Download the checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Update path in config.yaml
```

### Dataset Path Issues
- Ensure images are in `data/raw/images/`
- Ensure annotations are in `data/raw/annotations/`
- Update paths in `scripts/02_preprocess_data.py` if needed

## 📚 References

- **VisDrone Dataset**: [http://aiskyeye.com/](http://aiskyeye.com/)
- **YOLO**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **RT-DETR**: [https://arxiv.org/abs/2304.08069](https://arxiv.org/abs/2304.08069)
- **SAM**: [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

For hackathon use: Feel free to extend with additional models, metrics, or visualizations!

## ✨ Acknowledgments

- VisDrone team for the dataset
- Ultralytics for YOLO and RT-DETR implementations
- Meta AI for Segment Anything Model