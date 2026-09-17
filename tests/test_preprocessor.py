"""
Unit Tests for ImagePreprocessor Module.
Validates grayscale conversion, CLAHE equalization, resizing, and normalization.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.preprocessor import ImagePreprocessor


def test_preprocessor_shape_and_range():
    """Checks output tensor shape and normalization range."""
    preprocessor = ImagePreprocessor(target_size=(48, 48), use_clahe=True)
    
    # 200x200 BGR dummy image
    dummy_img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    crop, tensor = preprocessor.crop_and_preprocess(dummy_img, bbox=(20, 20, 100, 100))

    assert crop.shape == (48, 48), f"Expected crop shape (48, 48), got {crop.shape}"
    assert tensor.shape == (1, 1, 48, 48), f"Expected tensor shape (1, 1, 48, 48), got {tensor.shape}"
    assert isinstance(tensor, torch.Tensor)
    
    # Check bounds [-1, 1]
    assert tensor.min().item() >= -1.0
    assert tensor.max().item() <= 1.0


def test_preprocessor_invalid_roi():
    """Checks that invalid or empty crops raise ValueError."""
    preprocessor = ImagePreprocessor()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        preprocessor.crop_and_preprocess(dummy_img, bbox=(0, 0, 0, 0))
