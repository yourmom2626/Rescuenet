"""
Prediction and inference code for U-Net segmentation model.
Loads trained model and generates predictions on new images.
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import os
from pathlib import Path


class SegmentationPredictor:
    """Load trained U-Net model and generate predictions."""
    
    def __init__(self, model_path='unet_model.keras'):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to trained .keras model file
        """
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.image_size = (256, 256)
        print("✓ Model loaded successfully!")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image for prediction.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Preprocessed image array
        """
        # Read image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        original_shape = image.shape[:2]
        image_resized = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        
        # Normalize
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return image_normalized, image_resized, original_shape
    
    def predict(self, image_path, confidence_threshold=0.5):
        """
        Generate segmentation mask for an image.
        
        Args:
            image_path: Path to input image
            confidence_threshold: Threshold for binary segmentation
        
        Returns:
            Dict with predictions and original image
        """
        # Preprocess
        image_normalized, image_resized, original_shape = self.preprocess_image(image_path)
        
        # Add batch dimension
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Predict
        prediction = self.model.predict(image_batch, verbose=0)
        
        # Get segmentation mask (class 1 probability)
        mask_prob = prediction[0, :, :, 1]
        mask_binary = (mask_prob > confidence_threshold).astype(np.uint8)
        
        # Resize mask back to original image size
        mask_original = cv2.resize(
            mask_binary,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        
        return {
            'image': image_resized,
            'mask_probability': mask_prob,
            'mask_binary': mask_binary,
            'mask_original': mask_original,
            'original_shape': original_shape
        }
    
    def visualize_prediction(self, prediction, output_path=None):
        """
        Visualize segmentation result.
        
        Args:
            prediction: Prediction dict from predict()
            output_path: Optional path to save visualization
        
        Returns:
            Composite visualization image
        """
        image = prediction['image']
        mask = prediction['mask_binary']
        
        # Create overlay
        overlay = image.copy()
        overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
        
        # Concatenate original, mask, and overlay
        visualization = np.hstack([
            image,
            np.stack([mask * 255] * 3, axis=2).astype(np.uint8),
            overlay.astype(np.uint8)
        ])
        
        if output_path:
            # Convert RGB to BGR for OpenCV
            visualization_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), visualization_bgr)
            print(f"✓ Visualization saved to {output_path}")
        
        return visualization
    
    def batch_predict(self, image_dir, output_dir=None, confidence_threshold=0.5):
        """
        Generate predictions for all images in a directory.
        
        Args:
            image_dir: Directory containing input images
            output_dir: Optional directory to save results
            confidence_threshold: Threshold for segmentation
        
        Returns:
            List of prediction results
        """
        image_dir = Path(image_dir)
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            masks_dir = output_dir / 'masks'
            masks_dir.mkdir(exist_ok=True)
            viz_dir = output_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
        
        results = []
        
        # Process all images
        image_files = sorted([
            f for f in image_dir.glob('*') 
            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']
        ])
        
        print(f"\nProcessing {len(image_files)} images...")
        
        for i, image_file in enumerate(image_files):
            prediction = self.predict(image_file, confidence_threshold)
            results.append({
                'filename': image_file.name,
                'prediction': prediction
            })
            
            if output_dir:
                # Save mask
                mask_file = masks_dir / f"{image_file.stem}_mask.png"
                cv2.imwrite(str(mask_file), prediction['mask_original'] * 255)
                
                # Save visualization
                self.visualize_prediction(
                    prediction,
                    viz_dir / f"{image_file.stem}_viz.png"
                )
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(image_files)} images")
        
        print(f"✓ Batch prediction complete!")
        return results


def main():
    """Example usage of segmentation predictor."""
    
    # Initialize predictor with trained model
    predictor = SegmentationPredictor('unet_model.keras')
    
    # Create sample image for demonstration
    if os.path.exists('./data/images'):
        # Predict on first sample from dataset
        sample_image = './data/images/image_0000.png'
        
        if os.path.exists(sample_image):
            print("\nGenerating prediction on sample image...")
            prediction = predictor.predict(sample_image)
            
            print(f"Image shape: {prediction['image'].shape}")
            print(f"Mask shape: {prediction['mask_binary'].shape}")
            print(f"Infected pixels: {(prediction['mask_binary'] == 1).sum()}")
            
            # Visualize and save
            predictor.visualize_prediction(
                prediction,
                output_path='prediction_example.png'
            )
            print("✓ Example prediction saved!")


if __name__ == '__main__':
    main()
