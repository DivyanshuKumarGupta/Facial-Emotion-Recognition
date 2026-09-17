"""
Face Image Preprocessor Module.
Handles grayscale conversion, CLAHE illumination normalization,
resizing to 48x48, and conversion to PyTorch normalized tensors.
"""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Standardizes face image regions into model-ready tensors.
    """
    def __init__(self, target_size: Tuple[int, int] = (48, 48), use_clahe: bool = True):
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None

    def crop_and_preprocess(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Crops face region if bbox is provided, applies CLAHE, and returns
        both the preprocessed 48x48 numpy array and normalized PyTorch tensor (1, 1, 48, 48).
        """
        if bbox is not None:
            x, y, w, h = bbox
            face_roi = frame[y:y+h, x:x+w]
        else:
            face_roi = frame

        if face_roi is None or face_roi.size == 0:
            raise ValueError("Input ROI is empty or invalid.")

        # Convert to Grayscale
        if len(face_roi.shape) == 3 and face_roi.shape[2] == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi.copy()

        # Illumination normalization with CLAHE
        if self.clahe is not None:
            gray = self.clahe.apply(gray)

        # Resize to FER-2013 standard dimensions (48x48)
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

        # Normalize pixel values to [-1, 1]
        norm_arr = (resized.astype(np.float32) / 127.5) - 1.0

        # Transform to PyTorch Tensor (B=1, C=1, H=48, W=48)
        tensor = torch.from_numpy(norm_arr).unsqueeze(0).unsqueeze(0).float()

        return resized, tensor
