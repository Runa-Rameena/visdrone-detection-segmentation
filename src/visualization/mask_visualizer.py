import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt

class MaskVisualizer:
    """Visualize segmentation masks"""
    
    def __init__(self):
        """Initialize mask visualizer"""
        np.random.seed(42)
    
    @staticmethod
    def _generate_color():
        """Generate a random color"""
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def visualize_masks(self, image_path: str, masks: List[Dict], 
                       save_path: str = None, alpha: float = 0.5) -> np.ndarray:
        """
        Visualize segmentation masks on image
        
        Args:
            image_path: Path to input image
            masks: List of mask dictionaries from SAM
            save_path: Optional path to save visualization
            alpha: Transparency for mask overlay
            
        Returns:
            Image with visualized masks
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None
        
        overlay = image.copy()
        
        # Draw each mask
        for mask_dict in masks:
            mask = mask_dict['mask']
            color = self._generate_color()
            
            # Create colored mask
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = color
            
            # Blend with overlay
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, alpha, 0)
            
            # Draw contours for better visibility
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, overlay)
        
        return overlay
    
    def create_mask_comparison(self, image_path: str, bbox_detections: List[Dict], 
                              sam_masks: List[Dict], save_path: str = None):
        """
        Create side-by-side comparison of bboxes and masks
        
        Args:
            image_path: Path to input image
            bbox_detections: List of bounding box detections
            sam_masks: List of SAM masks
            save_path: Optional path to save comparison
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Bounding boxes
        bbox_image = image.copy()
        for det in bbox_detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            color = self._generate_color()
            cv2.rectangle(bbox_image, (x1, y1), (x2, y2), color, 2)
        
        bbox_image_rgb = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB)
        axes[1].imshow(bbox_image_rgb)
        axes[1].set_title(f'Bounding Boxes ({len(bbox_detections)} objects)')
        axes[1].axis('off')
        
        # Segmentation masks
        mask_image = self.visualize_masks(image_path, sam_masks)
        mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        axes[2].imshow(mask_image_rgb)
        axes[2].set_title(f'Segmentation Masks ({len(sam_masks)} objects)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def visualize_mask_quality(self, image_path: str, masks: List[Dict], 
                              save_path: str = None):
        """
        Visualize individual masks with quality scores
        
        Args:
            image_path: Path to input image
            masks: List of mask dictionaries with scores
            save_path: Optional path to save visualization
        """
        if len(masks) == 0:
            print("No masks to visualize")
            return
        
        # Sort masks by score
        sorted_masks = sorted(masks, key=lambda x: x.get('score', 0), reverse=True)
        
        # Show top masks
        num_masks = min(len(sorted_masks), 9)
        rows = int(np.ceil(num_masks / 3))
        
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for idx, mask_dict in enumerate(sorted_masks[:num_masks]):
            row = idx // 3
            col = idx % 3
            
            mask = mask_dict['mask']
            score = mask_dict.get('score', 0)
            
            # Create mask overlay
            masked_image = image_rgb.copy()
            masked_image[~mask] = masked_image[~mask] * 0.3  # Darken background
            
            axes[row, col].imshow(masked_image)
            axes[row, col].set_title(f'Mask {idx+1} (Score: {score:.3f})')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for idx in range(num_masks, rows * 3):
            row = idx // 3
            col = idx % 3
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_overlay_comparison(self, image_path: str, 
                                 detections_dict: Dict[str, List], 
                                 sam_masks: List[Dict],
                                 save_path: str = None):
        """
        Create comprehensive comparison of all models
        
        Args:
            image_path: Path to input image
            detections_dict: Dictionary of model detections
            sam_masks: SAM segmentation masks
            save_path: Optional path to save comparison
        """
        num_models = len(detections_dict)
        fig, axes = plt.subplots(1, num_models + 2, figsize=(6 * (num_models + 2), 6))
        
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Original
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Each detection model
        for idx, (model_name, detections) in enumerate(detections_dict.items(), 1):
            det_image = image.copy()
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(det_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            det_image_rgb = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(det_image_rgb)
            axes[idx].set_title(f'{model_name}\n({len(detections)} objects)')
            axes[idx].axis('off')
        
        # SAM masks
        mask_image = self.visualize_masks(image_path, sam_masks)
        mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
        axes[-1].imshow(mask_image_rgb)
        axes[-1].set_title(f'SAM Segmentation\n({len(sam_masks)} masks)')
        axes[-1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def main():
    """Test mask visualizer"""
    visualizer = MaskVisualizer()
    print("Mask Visualizer initialized")

if __name__ == "__main__":
    main()