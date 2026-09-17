"""
Face Detection Module using OpenCV Haar Cascade Classifier.
Provides robust real-time face detection with margin padding and spatial filtering.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """
    Detects frontal human faces within BGR or grayscale images.
    """
    def __init__(
        self,
        scale_factor: float = 1.15,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (40, 40)
    ):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load OpenCV's built-in Haar Cascade for frontal faces
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load OpenCV face cascade from {cascade_path}")

    def detect_faces(
        self,
        frame: np.ndarray,
        expand_ratio: float = 0.1
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces and returns a list of bounding boxes (x, y, w, h).
        
        Args:
            frame: Input image (BGR or Grayscale).
            expand_ratio: Optional bounding box margin expansion ratio.
            
        Returns:
            List of tuples: (x, y, width, height)
        """
        if frame is None or frame.size == 0:
            return []
            
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        h_img, w_img = frame.shape[:2]
        padded_faces = []
        
        for (x, y, w, h) in faces:
            # Apply padding margin around detected face
            dx = int(w * expand_ratio)
            dy = int(h * expand_ratio)
            
            x1 = max(0, x - dx)
            y1 = max(0, y - dy)
            x2 = min(w_img, x + w + dx)
            y2 = min(h_img, y + h + dy)
            
            padded_faces.append((x1, y1, x2 - x1, y2 - y1))
            
        return padded_faces
