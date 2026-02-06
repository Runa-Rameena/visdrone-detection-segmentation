import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import json
from pathlib import Path

class DetectionMetrics:
    """Calculate evaluation metrics for object detection"""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize metrics calculator
        
        Args:
            iou_threshold: IOU threshold for matching predictions with ground truth
        """
        self.iou_threshold = iou_threshold
    
    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two boxes
        
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_predictions(self, predictions: List[Dict], ground_truths: List[Dict], ignore_class: bool = False) -> Tuple[List, List, List]:
        """
        Match predictions with ground truth boxes
        
        Args:
            predictions: List of predicted boxes with confidence scores
            ground_truths: List of ground truth boxes
            ignore_class: If True, match boxes regardless of class label (useful for comparing across different taxonomies)
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        # Sort predictions by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        matched_gt = set()
        true_positives = []
        false_positives = []
        
        for pred in predictions:
            pred_box = pred['bbox']
            pred_class = pred['class']
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                    
                gt_box = gt['bbox']
                gt_class = gt['class']
                
                # Only match same class (unless ignore_class is True)
                if not ignore_class and pred_class != gt_class:
                    continue
                
                iou = self.calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is good enough
            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                true_positives.append({
                    'prediction': pred,
                    'ground_truth': ground_truths[best_gt_idx],
                    'iou': best_iou
                })
                matched_gt.add(best_gt_idx)
            else:
                false_positives.append(pred)
        
        # Unmatched ground truths are false negatives
        false_negatives = [gt for idx, gt in enumerate(ground_truths) if idx not in matched_gt]
        
        return true_positives, false_positives, false_negatives
    
    def calculate_precision_recall(self, true_positives: int, false_positives: int, false_negatives: int) -> Tuple[float, float]:
        """
        Calculate precision and recall
        
        Args:
            true_positives: Number of true positives
            false_positives: Number of false positives
            false_negatives: Number of false negatives
            
        Returns:
            Tuple of (precision, recall)
        """
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        return precision, recall
    
    def calculate_ap(self, precisions: List[float], recalls: List[float]) -> float:
        """
        Calculate Average Precision (AP) using 11-point interpolation
        
        Args:
            precisions: List of precision values
            recalls: List of recall values
            
        Returns:
            Average Precision
        """
        if len(precisions) == 0 or len(recalls) == 0:
            return 0.0
        
        # Sort by recall
        sorted_indices = np.argsort(recalls)
        recalls = np.array(recalls)[sorted_indices]
        precisions = np.array(precisions)[sorted_indices]
        
        # 11-point interpolation
        ap = 0.0
        for threshold in np.linspace(0, 1, 11):
            precisions_above = precisions[recalls >= threshold]
            if len(precisions_above) > 0:
                ap += np.max(precisions_above)
        
        return ap / 11.0
    
    def evaluate_dataset(self, predictions_by_id: Dict[str, List[Dict]], 
                         ground_truths_by_id: Dict[str, List[Dict]], 
                         ignore_class_mismatch: bool = True) -> Dict:
        """
        Evaluate entire dataset
        
        Args:
            predictions_by_id: Dict mapping image_id -> list of predictions
            ground_truths_by_id: Dict mapping image_id -> list of ground truth boxes
            ignore_class_mismatch: If True, match boxes regardless of class (for pretrained models on different datasets)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'precisions': [], 'recalls': []})
        
        # Iterate through all ground truth images
        for image_id, ground_truths in ground_truths_by_id.items():
            predictions = predictions_by_id.get(image_id, [])
            
            tp, fp, fn = self.match_predictions(predictions, ground_truths, ignore_class=ignore_class_mismatch)
            
            total_tp += len(tp)
            total_fp += len(fp)
            total_fn += len(fn)
            
            # Per-class metrics
            for t in tp:
                cls = t['prediction']['class']
                class_metrics[cls]['tp'] += 1
            
            for f in fp:
                cls = f['class']
                class_metrics[cls]['fp'] += 1
            
            for f in fn:
                cls = f['class']
                class_metrics[cls]['fn'] += 1
        
        # Calculate overall metrics
        precision, recall = self.calculate_precision_recall(total_tp, total_fp, total_fn)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate per-class metrics
        class_results = {}
        aps = []
        
        for cls, metrics in class_metrics.items():
            cls_precision, cls_recall = self.calculate_precision_recall(
                metrics['tp'], metrics['fp'], metrics['fn']
            )
            class_results[cls] = {
                'precision': cls_precision,
                'recall': cls_recall,
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn']
            }
            
            if cls_precision > 0 or cls_recall > 0:
                aps.append(cls_precision)  # Simplified AP calculation
        
        mAP = np.mean(aps) if len(aps) > 0 else 0.0
        
        return {
            'overall': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'mAP': mAP,
                'total_tp': total_tp,
                'total_fp': total_fp,
                'total_fn': total_fn
            },
            'per_class': class_results
        }
    
    def save_metrics(self, metrics: Dict, output_path: str):
        """Save metrics to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Metrics saved to {output_path}")

def main():
    """Test metrics calculation"""
    metrics = DetectionMetrics(iou_threshold=0.5)
    
    # Example predictions and ground truths
    predictions = [
        {'bbox': [10, 10, 50, 50], 'confidence': 0.9, 'class': 1},
        {'bbox': [60, 60, 100, 100], 'confidence': 0.8, 'class': 2},
    ]
    
    ground_truths = [
        {'bbox': [12, 12, 48, 48], 'class': 1},
        {'bbox': [65, 65, 95, 95], 'class': 2},
    ]
    
    tp, fp, fn = metrics.match_predictions(predictions, ground_truths)
    print(f"True Positives: {len(tp)}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")

if __name__ == "__main__":
    main()