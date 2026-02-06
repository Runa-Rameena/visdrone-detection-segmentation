import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class VisDroneParser:
    """Parse VisDrone dataset annotations"""
    
    def __init__(self, annotations_dir: str):
        self.annotations_dir = Path(annotations_dir)
        
    def parse_annotation(self, annotation_file: str) -> List[Dict]:
        """
        Parse a single VisDrone annotation file
        
        VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        
        Returns:
            List of dictionaries containing bbox and class info
        """
        annotations = []
        
        if not os.path.exists(annotation_file):
            return annotations
            
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(',')
            if len(parts) < 6:
                continue
                
            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_width = int(parts[2])
            bbox_height = int(parts[3])
            score = int(parts[4])  # Usually 0 for ground truth
            object_category = int(parts[5])
            truncation = int(parts[6]) if len(parts) > 6 else 0
            occlusion = int(parts[7]) if len(parts) > 7 else 0
            
            # Skip ignored regions (class 0) and invalid boxes
            if object_category == 0 or bbox_width <= 0 or bbox_height <= 0:
                continue
            
            # Convert to [x1, y1, x2, y2] format
            x1 = bbox_left
            y1 = bbox_top
            x2 = bbox_left + bbox_width
            y2 = bbox_top + bbox_height
            
            annotations.append({
                'bbox': [x1, y1, x2, y2],
                'class': object_category,
                'truncation': truncation,
                'occlusion': occlusion,
                'area': bbox_width * bbox_height
            })
        
        return annotations
    
    def get_all_annotations(self) -> Dict[str, List[Dict]]:
        """
        Parse all annotation files in the directory
        
        Returns:
            Dictionary mapping image_id to list of annotations
        """
        all_annotations = {}
        
        for ann_file in self.annotations_dir.glob('*.txt'):
            image_id = ann_file.stem
            annotations = self.parse_annotation(str(ann_file))
            all_annotations[image_id] = annotations
        
        return all_annotations
    
    @staticmethod
    def convert_to_yolo_format(annotations: List[Dict], img_width: int, img_height: int) -> List[str]:
        """
        Convert annotations to YOLO format (normalized xywh)
        
        YOLO format: <class> <x_center> <y_center> <width> <height>
        All values normalized to [0, 1]
        """
        yolo_annotations = []
        
        for ann in annotations:
            x1, y1, x2, y2 = ann['bbox']
            class_id = ann['class'] - 1  # YOLO classes start from 0
            
            # Calculate center and dimensions
            x_center = (x1 + x2) / 2.0 / img_width
            y_center = (y1 + y2) / 2.0 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        return yolo_annotations
    
    @staticmethod
    def is_small_object(annotation: Dict, threshold: int = 32) -> bool:
        """
        Determine if an object is small based on area threshold
        
        Args:
            annotation: Annotation dictionary
            threshold: Pixel threshold for small objects (default: 32x32 = 1024 pixels)
        """
        return annotation['area'] < (threshold * threshold)