import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

class BBoxVisualizer:
    """Visualize bounding box detections"""
    
    def __init__(self, class_names: Dict[int, str] = None):
        """
        Initialize visualizer
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        self.class_names = class_names or {}
        self.colors = self._generate_colors(len(self.class_names))
    
    @staticmethod
    def _generate_colors(num_classes: int) -> np.ndarray:
        """Generate distinct colors for each class"""
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_classes + 1, 3), dtype=np.uint8)
        return colors
    
    def draw_bbox(self, image: np.ndarray, bbox: List[float], class_id: int, 
                  confidence: float = None, thickness: int = 2) -> np.ndarray:
        """
        Draw a single bounding box on image
        
        Args:
            image: Input image
            bbox: [x1, y1, x2, y2]
            class_id: Class ID
            confidence: Optional confidence score
            thickness: Box thickness
            
        Returns:
            Image with drawn bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for this class
        color = tuple(map(int, self.colors[class_id % len(self.colors)]))
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Create label
        label = self.class_names.get(class_id, f"Class {class_id}")
        if confidence is not None:
            label = f"{label} {confidence:.2f}"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_w, label_h = label_size
        
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def visualize_detections(self, image_path: str, detections: List[Dict], 
                           save_path: str = None) -> np.ndarray:
        """
        Visualize all detections on an image
        
        Args:
            image_path: Path to input image
            detections: List of detection dictionaries
            save_path: Optional path to save visualization
            
        Returns:
            Image with visualized detections
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        # Draw each detection
        for det in detections:
            bbox = det['bbox']
            class_id = det['class']
            confidence = det.get('confidence', None)
            
            image = self.draw_bbox(image, bbox, class_id, confidence)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, image)
        
        return image
    
    def create_comparison_grid(self, image_path: str, detections_dict: Dict[str, List[Dict]], 
                              save_path: str = None):
        """
        Create a grid comparing detections from multiple models
        
        Args:
            image_path: Path to input image
            detections_dict: Dictionary mapping model names to detection lists
            save_path: Optional path to save comparison
        """
        num_models = len(detections_dict)
        
        fig, axes = plt.subplots(1, num_models + 1, figsize=(6 * (num_models + 1), 6))
        
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Show original
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Show each model's detections
        for idx, (model_name, detections) in enumerate(detections_dict.items(), 1):
            viz_image = self.visualize_detections(image_path, detections)
            viz_image_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
            
            axes[idx].imshow(viz_image_rgb)
            axes[idx].set_title(f'{model_name}\n({len(detections)} detections)')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confidence_distribution(self, all_detections: List[Dict], save_path: str = None):
        """
        Plot confidence score distribution
        
        Args:
            all_detections: List of all detection results
            save_path: Optional path to save plot
        """
        confidences = []
        for result in all_detections:
            for det in result.get('detections', []):
                confidences.append(det.get('confidence', 0))
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Detection Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_class_distribution(self, all_detections: List[Dict], save_path: str = None):
        """
        Plot class distribution
        
        Args:
            all_detections: List of all detection results
            save_path: Optional path to save plot
        """
        class_counts = {}
        for result in all_detections:
            for det in result.get('detections', []):
                class_id = det['class']
                class_name = self.class_names.get(class_id, f"Class {class_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts, edgecolor='black', alpha=0.7)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Detection Class Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    """Test bbox visualizer"""
    # Example usage
    class_names = {
        1: "pedestrian",
        2: "car",
        3: "bicycle"
    }
    
    visualizer = BBoxVisualizer(class_names)
    
    # Example detections
    detections = [
        {'bbox': [10, 10, 50, 50], 'confidence': 0.9, 'class': 1},
        {'bbox': [60, 60, 100, 100], 'confidence': 0.85, 'class': 2},
    ]
    
    print("BBox Visualizer initialized")
    print(f"Class names: {class_names}")

if __name__ == "__main__":
    main()