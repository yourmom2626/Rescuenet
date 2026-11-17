"""
Training script for U-Net segmentation model.
Trains on image-mask pairs and saves the model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import os
import sys

from unet_model import unet_model
from dataset import SyntheticSegmentationDataset, load_dataset


def train_unet(epochs=50, batch_size=16, validation_split=0.2, model_save_path='unet_model.keras'):
    """
    Train U-Net model on image-mask pairs.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        model_save_path: Path to save trained model
    """
    
    print("=" * 50)
    print("U-Net Training Pipeline")
    print("=" * 50)
    
    # Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    dataset_gen = SyntheticSegmentationDataset(
        num_samples=100,
        image_size=(256, 256),
        num_classes=2
    )
    images, masks = dataset_gen.generate(save_dir='./data')
    
    # Split into train and validation sets
    print("\n2. Splitting dataset...")
    x_train, x_val, y_train, y_val = train_test_split(
        images, masks, test_size=validation_split, random_state=42
    )
    print(f"Training set: {x_train.shape}")
    print(f"Validation set: {x_val.shape}")
    
    # Build model
    print("\n3. Building U-Net model...")
    model = unet_model(input_shape=(256, 256, 3), num_classes=2)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.MeanIoU(num_classes=2)]
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("\n4. Training model...")
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print(f"\n5. Saving model to {model_save_path}...")
    model.save(model_save_path)
    print(f"✓ Model saved successfully!")
    
    # Evaluate on validation set
    print("\n6. Evaluating model...")
    val_loss, val_acc, val_iou = model.evaluate(x_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation IoU: {val_iou:.4f}")
    
    return model, history


def main():
    """Main training function."""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train model
    model, history = train_unet(
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        model_save_path='unet_model.keras'
    )
    
    print("\n" + "=" * 50)
    print("Training complete! Model saved as 'unet_model.keras'")
    print("=" * 50)


if __name__ == '__main__':
    main()
