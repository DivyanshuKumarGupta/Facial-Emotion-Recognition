"""
FER-2013 Dataset Ingestion and Preprocessing Pipeline
Provides PyTorch Dataset implementations, on-the-fly augmentations,
and synthetic split generation for validation/testing.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FER2013Dataset(Dataset):
    """
    Dataset loader for FER-2013 formatted data.
    Supports loading directly from CSV (Usage: Training, PublicTest, PrivateTest)
    or numpy arrays.
    """
    def __init__(
        self,
        pixels: np.ndarray,
        labels: np.ndarray,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            pixels: Array of shape (N, 48, 48) containing uint8 pixel values.
            labels: Array of shape (N,) containing integer labels (0-6).
            transform: Optional torchvision transformations.
        """
        self.pixels = pixels.astype(np.uint8)
        self.labels = labels.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_arr = self.pixels[idx]
        label = self.labels[idx]
        
        img = Image.fromarray(img_arr, mode='L') # 1-channel Grayscale
        
        if self.transform:
            img = self.transform(img)
        else:
            # Default tensor conversion and normalization to [-1, 1]
            t = transforms.ToTensor()
            img = t(img)
            img = (img - 0.5) / 0.5
            
        return img, torch.tensor(label, dtype=torch.long)


def get_default_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Returns data augmentation pipeline for training or standard normalization for validation.
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(48, scale=(0.85, 1.15)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


def load_fer2013_from_csv(
    csv_path: Path,
    usage_filter: Optional[str] = None,
    transform: Optional[transforms.Compose] = None
) -> FER2013Dataset:
    """
    Loads samples from standard Kaggle FER-2013 CSV file ('emotion', 'pixels', 'Usage').
    """
    df = pd.read_csv(csv_path)
    if usage_filter and "Usage" in df.columns:
        df = df[df["Usage"] == usage_filter].reset_index(drop=True)
        
    labels = df["emotion"].values
    
    # Parse whitespace-separated string pixels
    pixels_list = []
    for pixel_seq in df["pixels"]:
        arr = np.fromstring(pixel_seq, sep=" ", dtype=np.uint8).reshape(48, 48)
        pixels_list.append(arr)
        
    pixels = np.stack(pixels_list, axis=0)
    return FER2013Dataset(pixels=pixels, labels=labels, transform=transform)


def create_benchmark_split(
    num_samples_per_class: int = 40,
    seed: int = 42
) -> Tuple[FER2013Dataset, FER2013Dataset]:
    """
    Generates deterministic benchmark dataset for offline CI/CD and validation.
    Synthesizes facial spatial patterns for all 7 classes.
    """
    np.random.seed(seed)
    total_samples = num_samples_per_class * 7
    pixels = np.zeros((total_samples, 48, 48), dtype=np.uint8)
    labels = np.zeros(total_samples, dtype=np.int64)
    
    idx = 0
    for emotion_idx in range(7):
        for _ in range(num_samples_per_class):
            # Base face circle pattern
            y, x = np.ogrid[:48, :48]
            mask = (x - 24)**2 + (y - 24)**2 <= 20**2
            face = np.full((48, 48), 120, dtype=np.uint8)
            face[mask] = 175
            
            # Eyes
            face[16:20, 16:20] = 50
            face[16:20, 28:32] = 50
            
            # Emotion-specific mouth expressions
            if emotion_idx == 3: # Happy (smile arc)
                face[32:36, 18:30] = 230
            elif emotion_idx == 4: # Sad (downward frown)
                face[34:38, 18:30] = 40
            elif emotion_idx == 5: # Surprise (round mouth)
                face[30:38, 22:26] = 20
            elif emotion_idx == 0: # Angry (angled brow lines)
                face[13:16, 15:22] = 30
                face[13:16, 26:33] = 30
            else: # Neutral / Disgust / Fear
                face[34:36, 18:30] = 100
                
            # Add subtle natural texture noise
            noise = np.random.randint(-15, 15, (48, 48))
            face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            pixels[idx] = face
            labels[idx] = emotion_idx
            idx += 1
            
    # Shuffle
    perm = np.random.permutation(total_samples)
    pixels, labels = pixels[perm], labels[perm]
    
    split = int(0.8 * total_samples)
    train_ds = FER2013Dataset(pixels[:split], labels[:split], transform=get_default_transforms(True))
    val_ds = FER2013Dataset(pixels[split:], labels[split:], transform=get_default_transforms(False))
    
    return train_ds, val_ds


def get_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """Wraps datasets into PyTorch DataLoader instances."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
