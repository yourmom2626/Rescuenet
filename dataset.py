"""
Synthetic dataset generator for U-Net training.
Creates paired images and segmentation masks.
"""

import numpy as np
import cv2
from tensorflow import keras
import os


class SyntheticSegmentationDataset:
    """Generate synthetic image-mask pairs for training."""
    
    def __init__(self, num_samples=100, image_size=(256, 256), num_classes=2):
        """
        Initialize dataset generator.
        
        Args:
            num_samples: Number of training samples to generate
            image_size: (height, width) of images
            num_classes: Number of segmentation classes
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        
    def _generate_synthetic_image(self):
        """Generate a random synthetic image with shapes."""
        image = np.ones((*self.image_size, 3), dtype=np.uint8) * 200
        
        # Add some random shapes and noise
        num_circles = np.random.randint(2, 5)
        for _ in range(num_circles):
            x = np.random.randint(20, self.image_size[1] - 20)
            y = np.random.randint(20, self.image_size[0] - 20)
            radius = np.random.randint(10, 40)
            color = tuple(np.random.randint(50, 200, 3).tolist())
            cv2.circle(image, (x, y), radius, color, -1)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def _generate_synthetic_mask(self):
        """Generate a segmentation mask with two classes."""
        mask = np.zeros(self.image_size, dtype=np.uint8)
        
        # Create some infected regions (class 1)
        num_regions = np.random.randint(1, 4)
        for _ in range(num_regions):
            x = np.random.randint(30, self.image_size[1] - 30)
            y = np.random.randint(30, self.image_size[0] - 30)
            radius = np.random.randint(15, 50)
            cv2.circle(mask, (x, y), radius, 1, -1)
        
        return mask
    
    def generate(self, save_dir='./data'):
        """
        Generate and save synthetic dataset.
        
        Returns:
            Tuple of (images_array, masks_array)
        """
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/images', exist_ok=True)
        os.makedirs(f'{save_dir}/masks', exist_ok=True)
        
        images = []
        masks = []
        
        for i in range(self.num_samples):
            image = self._generate_synthetic_image()
            mask = self._generate_synthetic_mask()
            
            images.append(image)
            masks.append(mask)
            
            # Save individual files
            cv2.imwrite(f'{save_dir}/images/image_{i:04d}.png', image)
            cv2.imwrite(f'{save_dir}/masks/mask_{i:04d}.png', mask * 255)
            
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{self.num_samples} samples")
        
        # Convert to numpy arrays and normalize
        images = np.array(images, dtype=np.float32) / 255.0
        masks = np.array(masks, dtype=np.float32)
        
        # Convert masks to one-hot encoding
        masks_onehot = np.zeros((*masks.shape, self.num_classes), dtype=np.float32)
        masks_onehot[..., 0] = 1.0 - masks
        masks_onehot[..., 1] = masks
        
        print(f"\nDataset generation complete!")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape (one-hot): {masks_onehot.shape}")
        
        return images, masks_onehot


def load_dataset(data_dir='./data', num_classes=2):
    """
    Load generated dataset from disk.
    
    Args:
        data_dir: Directory containing 'images' and 'masks' subdirectories
        num_classes: Number of segmentation classes
    
    Returns:
        Tuple of (images_array, masks_array)
    """
    images = []
    masks = []
    
    image_dir = f'{data_dir}/images'
    mask_dir = f'{data_dir}/masks'
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    for img_file in image_files:
        img = cv2.imread(f'{image_dir}/{img_file}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img.astype(np.float32) / 255.0)
        
        mask_file = img_file.replace('image_', 'mask_')
        mask = cv2.imread(f'{mask_dir}/{mask_file}', cv2.IMREAD_GRAYSCALE)
        masks.append(mask.astype(np.float32) / 255.0)
    
    images = np.array(images)
    masks = np.array(masks)
    
    # Convert masks to one-hot encoding
    masks_onehot = np.zeros((*masks.shape, num_classes), dtype=np.float32)
    masks_onehot[..., 0] = 1.0 - masks
    masks_onehot[..., 1] = masks
    
    return images, masks_onehot


if __name__ == "__main__":
    # Generate synthetic dataset
    dataset_gen = SyntheticSegmentationDataset(num_samples=100, image_size=(256, 256), num_classes=2)
    images, masks = dataset_gen.generate()
