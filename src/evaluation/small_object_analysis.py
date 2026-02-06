import numpy as np
from typing import List, Dict
from collections import defaultdict
import json
from pathlib import Path

class SmallObjectAnalysis:
    """Analyze model performance on small objects"""
    
    def __init__(self, small_threshold: int = 32, medium_threshold: int = 96):
        """
        Initialize small object analyzer
        
        Args:
            small_threshold: Maximum dimension for small objects (default: 32x32)
            medium_threshold: Maximum dimension for medium objects (default: 96x96)
        """
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold
    
    @staticmethod
    def get_bbox_size(bbox: List[float]) -> tuple[float, float]:
        """
        Calculate bounding box dimensions
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Tuple of (width, height)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return width, height
    
    @staticmethod
    def get_bbox_area(bbox: List[float]) -> float:
        """Calculate bounding box area"""
        width, height = SmallObjectAnalysis.get_bbox_size(bbox)
        return width * height
    
    def categorize_object(self, bbox: List[float]) -> str:
        """
        Categorize object size based on area
        
        Args:
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Size category: 'small', 'medium', or 'large'
        """
        area = self.get_bbox_area(bbox)
        small_area = self.small_threshold ** 2
        medium_area = self.medium_threshold ** 2
        
        if area < small_area:
            return 'small'
        elif area < medium_area:
            return 'medium'
        else:
            return 'large'
    
    def analyze_detections(self, detections: List[Dict]) -> Dict:
        """
        Analyze detections by object size
        
        Args:
            detections: List of detection results
            
        Returns:
            Dictionary containing size-based statistics
        """
        size_stats = defaultdict(lambda: {'count': 0, 'confidences': [], 'areas': []})
        
        for det in detections:
            bbox = det['bbox']
            confidence = det.get('confidence', 1.0)
            
            category = self.categorize_object(bbox)
            area = self.get_bbox_area(bbox)
            
            size_stats[category]['count'] += 1
            size_stats[category]['confidences'].append(confidence)
            size_stats[category]['areas'].append(area)
        
        # Calculate summary statistics
        summary = {}
        for category, stats in size_stats.items():
            if stats['count'] > 0:
                summary[category] = {
                    'count': stats['count'],
                    'mean_confidence': np.mean(stats['confidences']),
                    'std_confidence': np.std(stats['confidences']),
                    'mean_area': np.mean(stats['areas']),
                    'min_area': np.min(stats['areas']),
                    'max_area': np.max(stats['areas'])
                }
        
        return summary
    
    def analyze_dataset(self, all_predictions: List[Dict], all_ground_truths: List[Dict] = None) -> Dict:
        """
        Analyze entire dataset for small object performance
        
        Args:
            all_predictions: List of prediction results for all images or Dict mapping image_id -> detections
            all_ground_truths: Optional list of ground truth annotations or Dict mapping image_id -> annotations
            
        Returns:
            Dictionary containing comprehensive small object analysis
        """
        # Handle both list and dict formats
        if isinstance(all_predictions, dict):
            all_predictions = list(all_predictions.values())
        if isinstance(all_ground_truths, dict):
            all_ground_truths = list(all_ground_truths.values())
        
        # Analyze predictions
        # Ensure the stats container contains all expected list fields (confidences, areas, aspect_ratios)
        pred_stats = defaultdict(lambda: {"confidences": [], "areas": [], "aspect_ratios": []})
        
        for pred_result in all_predictions:
            # Handle both dict and list formats
            if isinstance(pred_result, dict):
                detections = pred_result.get('detections', [])
            else:
                detections = pred_result
            
            for det in detections:
                bbox = det['bbox']
                confidence = det.get('confidence', 1.0)
                category = self.categorize_object(bbox)
                
                pred_stats[category]['confidences'].append(confidence)
                pred_stats[category]['areas'].append(self.get_bbox_area(bbox))
                # safe aspect ratio calculation
                w, h = self.get_bbox_size(bbox)
                aspect = (w / h) if h > 0 else 0.0
                pred_stats[category]['aspect_ratios'].append(aspect)
        
        # Calculate prediction statistics
        prediction_summary = {}
        for category, stats in pred_stats.items():
            if len(stats.get('confidences', [])) > 0:
                prediction_summary[category] = {
                    'total_detections': int(len(stats.get('confidences', []))),
                    'mean_confidence': float(np.mean(stats.get('confidences', []))),
                    'std_confidence': float(np.std(stats.get('confidences', []))),
                    'mean_area': float(np.mean(stats.get('areas', []))) if len(stats.get('areas', [])) > 0 else 0.0,
                    'median_area': float(np.median(stats.get('areas', []))) if len(stats.get('areas', [])) > 0 else 0.0,
                    'mean_aspect_ratio': float(np.mean(stats.get('aspect_ratios', []))) if len(stats.get('aspect_ratios', [])) > 0 else 0.0
                }
        
        result = {
            'predictions': prediction_summary,
            'thresholds': {
                'small': f"< {self.small_threshold}x{self.small_threshold}",
                'medium': f"{self.small_threshold}x{self.small_threshold} to {self.medium_threshold}x{self.medium_threshold}",
                'large': f"> {self.medium_threshold}x{self.medium_threshold}"
            }
        }
        
        # Analyze ground truths if provided
        if all_ground_truths:
            gt_stats = defaultdict(int)
            
            for gt_result in all_ground_truths:
                # Handle both dict and list formats
                if isinstance(gt_result, dict):
                    annotations = gt_result.get('annotations', [])
                else:
                    annotations = gt_result
                
                for ann in annotations:
                    bbox = ann['bbox']
                    category = self.categorize_object(bbox)
                    gt_stats[category] += 1
            
            result['ground_truth'] = dict(gt_stats)
            
            # Calculate detection rates by size
            detection_rates = {}
            for category in gt_stats.keys():
                gt_count = gt_stats[category]
                pred_count = prediction_summary.get(category, {}).get('total_detections', 0)
                detection_rates[category] = {
                    'ground_truth_count': gt_count,
                    'detection_count': pred_count,
                    'detection_rate': pred_count / gt_count if gt_count > 0 else 0.0
                }
            
            result['detection_rates'] = detection_rates
        
        return result
    
    def compare_models(self, model_results: Dict[str, List[Dict]]) -> Dict:
        """
        Compare small object performance across multiple models
        
        Args:
            model_results: Dictionary mapping model names to their prediction results
            
        Returns:
            Comparative analysis dictionary
        """
        comparison = {}
        
        for model_name, predictions in model_results.items():
            analysis = self.analyze_dataset(predictions)
            comparison[model_name] = analysis['predictions']
        
        return {
            'models': comparison,
            'thresholds': {
                'small': f"< {self.small_threshold}x{self.small_threshold}",
                'medium': f"{self.small_threshold}x{self.small_threshold} to {self.medium_threshold}x{self.medium_threshold}",
                'large': f"> {self.medium_threshold}x{self.medium_threshold}"
            }
        }
    
    def save_analysis(self, analysis: Dict, output_path: str):
        """Save analysis to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Small object analysis saved to {output_path}")

def main():
    """Test small object analysis"""
    analyzer = SmallObjectAnalysis(small_threshold=32, medium_threshold=96)
    
    # Example detections
    detections = [
        {'bbox': [10, 10, 30, 30], 'confidence': 0.9, 'class': 1},  # Small
        {'bbox': [50, 50, 120, 120], 'confidence': 0.85, 'class': 2},  # Medium
        {'bbox': [200, 200, 350, 350], 'confidence': 0.95, 'class': 3},  # Large
    ]
    
    analysis = analyzer.analyze_detections(detections)
    print("Detection analysis by size:")
    for category, stats in analysis.items():
        print(f"{category}: {stats}")

if __name__ == "__main__":
    main()