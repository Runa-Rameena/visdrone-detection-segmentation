import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO
import torch
from tqdm import tqdm

from src.config import config

class YOLOModel:
    """Wrapper for YOLO model inference"""
    
    def __init__(self, model_name: str = 'yolov11n.pt', conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize YOLO model
        
        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt', 'yolov8s.pt')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def predict(self, image_path: str, save_path: str = None) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save visualization
            
        Returns:
            Dictionary containing detections and metadata
        """
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Extract detections
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': conf,
                'class': cls,
                'class_name': self.model.names[cls]
            })
        
        # Save visualization if requested
        if save_path:
            annotated = results.plot()
            cv2.imwrite(save_path, annotated)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'num_detections': len(detections)
        }
    
    def predict_batch(self, image_dir: str, output_dir: str = None, save_visualizations: bool = True) -> List[Dict]:
        """
        Run inference on a batch of images
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            save_visualizations: Whether to save annotated images
            
        Returns:
            List of detection results for all images
        """
        image_path = Path(image_dir)
        image_files = list(image_path.glob('*.jpg')) + list(image_path.glob('*.png'))
        
        if output_dir and save_visualizations:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        print(f"Running YOLO inference on {len(image_files)} images...")
        for img_file in tqdm(image_files):
            save_path = None
            if output_dir and save_visualizations:
                save_path = output_path / f"{img_file.stem}_yolo.jpg"
            
            result = self.predict(str(img_file), str(save_path) if save_path else None)
            all_results.append(result)
        
        return all_results
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': self.device,
            'num_classes': len(self.model.names),
            'class_names': self.model.names
        }

def main():
    """Test YOLO model"""
    # Load configuration
    model_name = config.get('models.yolo.model_name', 'yolov8n.pt')
    conf_threshold = config.get('models.yolo.conf_threshold', 0.25)
    iou_threshold = config.get('models.yolo.iou_threshold', 0.45)
    
    # Initialize model
    yolo = YOLOModel(model_name, conf_threshold, iou_threshold)
    
    # Test on processed test set
    test_images = config.dataset_processed_path / 'test' / 'images'
    output_dir = config.detections_path / 'yolo'
    
    if test_images.exists():
        results = yolo.predict_batch(str(test_images), str(output_dir))
        print(f"Processed {len(results)} images")
        print(f"Model info: {yolo.get_model_info()}")
    else:
        print(f"Test images directory not found: {test_images}")

if __name__ == "__main__":
    main()