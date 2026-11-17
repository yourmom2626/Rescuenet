"""
U-Net Architecture for Image Segmentation
Implements the U-Net model as described in Ronneberger et al. (2015)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def unet_model(input_shape=(256, 256, 3), num_classes=2):
    """
    Build a U-Net model for image segmentation.
    
    Args:
        input_shape: Tuple of (height, width, channels)
        num_classes: Number of output classes (2 for binary segmentation)
    
    Returns:
        Compiled U-Net model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder (Contraction path)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(2)(c4)
    
    # Bottleneck
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
    
    # Decoder (Expansion path)
    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
    
    u8 = layers.UpSampling2D(2)(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
    
    u9 = layers.UpSampling2D(2)(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
    
    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(c9)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


if __name__ == "__main__":
    # Build and display model
    model = unet_model()
    model.summary()
