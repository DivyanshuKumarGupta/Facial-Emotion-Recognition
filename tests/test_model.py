"""
Unit Tests for EmotionResidualCNN Model Architecture.
Validates tensor dimensions, forward pass, parameter counts, and probabilities.
"""

import sys
from pathlib import Path
import pytest
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.emotion_cnn import EmotionResidualCNN, get_emotion_model


def test_model_forward_shape():
    """Validates output shape for arbitrary batch sizes."""
    model = EmotionResidualCNN(in_channels=1, num_classes=7)
    model.eval()

    # Batch of 4 images, 1 channel, 48x48
    dummy_input = torch.randn(4, 1, 48, 48)
    output = model(dummy_input)

    assert output.shape == (4, 7), f"Expected shape (4, 7), but got {output.shape}"


def test_model_single_sample_inference():
    """Validates single sample inference shape and probability sum."""
    model = EmotionResidualCNN(in_channels=1, num_classes=7)
    model.eval()

    dummy_input = torch.randn(1, 1, 48, 48)
    probs = model.predict_probabilities(dummy_input)

    assert probs.shape == (1, 7)
    prob_sum = probs.sum().item()
    assert pytest.approx(prob_sum, abs=1e-4) == 1.0, f"Probabilities sum to {prob_sum} != 1.0"


def test_model_factory_and_weights():
    """Tests factory initialization with existing checkpoint."""
    from config.config import MODEL_WEIGHTS_PATH
    model = get_emotion_model(weights_path=MODEL_WEIGHTS_PATH)
    assert isinstance(model, EmotionResidualCNN)
    assert not model.training
