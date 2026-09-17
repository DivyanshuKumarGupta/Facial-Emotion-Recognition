"""
Deep Residual Convolutional Neural Network for Facial Emotion Recognition.
Optimized for 48x48 single-channel grayscale facial crops from the FER-2013 dataset.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Standard residual convolutional block with batch normalization and skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class EmotionResidualCNN(nn.Module):
    """
    Multi-stage Deep Residual CNN architecture for Facial Emotion Classification (7 classes).
    Designed to achieve high feature extraction capacity while maintaining low latency (< 10ms per face).
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 7, dropout_rate: float = 0.3):
        super(EmotionResidualCNN, self).__init__()
        
        # Initial Feature Extractor
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 48x48 -> 24x24
            nn.Dropout2d(p=dropout_rate * 0.5)
        )
        
        # Stage 1: 32 -> 64 channels
        self.stage1 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2), # 24x24 -> 12x12
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Stage 2: 64 -> 128 channels
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2), # 12x12 -> 6x6
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Stage 3: 128 -> 256 channels
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.AdaptiveAvgPool2d((2, 2)),          # 6x6 -> 2x2
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        logits = self.classifier(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes normalized class probabilities via Softmax."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


def get_emotion_model(
    weights_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    eval_mode: bool = True
) -> EmotionResidualCNN:
    """
    Factory function to initialize EmotionResidualCNN with optional pretrained weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = EmotionResidualCNN(in_channels=1, num_classes=7)
    
    if weights_path and Path(weights_path).exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    
    model.to(device)
    if eval_mode:
        model.eval()
        
    return model
