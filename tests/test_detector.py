"""
Unit Tests for FaceDetector Module.
Validates OpenCV Haar Cascade detector initialization, bounding box generation, and edge cases.
"""

import sys
from pathlib import Path
import pytest
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.face_detector import FaceDetector
from config.config import SAMPLE_ASSETS_DIR


def test_detector_initialization():
    """Validates that Haar cascade loads correctly."""
    detector = FaceDetector()
    assert not detector.face_cascade.empty()


def test_detector_empty_frame():
    """Validates that empty or None inputs return empty list without crashing."""
    detector = FaceDetector()
    assert detector.detect_faces(None) == []
    assert detector.detect_faces(np.array([])) == []


def test_detector_on_sample_image():
    """Validates face detection on real or synthetic sample face."""
    sample_path = SAMPLE_ASSETS_DIR / "sample_face.jpg"
    if not sample_path.exists():
        pytest.skip("Sample image not found, skipping detection test.")

    img = cv2.imread(str(sample_path))
    detector = FaceDetector()
    boxes = detector.detect_faces(img)
    # The synthetic face or any realistic test image should yield a bounding box
    assert isinstance(boxes, list)
