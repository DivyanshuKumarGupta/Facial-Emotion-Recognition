"""
High-Level Emotion Predictor Pipeline.
Orchestrates face detection, preprocessing, deep model inference,
and returns structured predictions with confidence metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from config.config import EMOTION_LABELS, MODEL_WEIGHTS_PATH
from src.models.emotion_cnn import EmotionResidualCNN, get_emotion_model
from src.pipeline.face_detector import FaceDetector
from src.pipeline.preprocessor import ImagePreprocessor


class EmotionPredictor:
    """
    End-to-end inference engine for facial emotion analysis.
    """
    def __init__(
        self,
        weights_path: Optional[Path] = MODEL_WEIGHTS_PATH,
        device: Optional[torch.device] = None,
        scale_factor: float = 1.15,
        min_neighbors: int = 5,
        temporal_smoothing: bool = True
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.detector = FaceDetector(scale_factor=scale_factor, min_neighbors=min_neighbors)
        self.preprocessor = ImagePreprocessor(target_size=(48, 48), use_clahe=True)
        self.model = get_emotion_model(weights_path=weights_path, device=self.device, eval_mode=True)
        
        self.temporal_smoothing = temporal_smoothing
        self._history: Dict[int, List[np.ndarray]] = {}

    def predict_frame(
        self,
        frame: np.ndarray,
        direct_crop: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Processes a single frame and outputs predictions for all detected faces.
        
        Args:
            frame: BGR numpy image frame.
            direct_crop: If True, treats entire frame as already cropped face.
            
        Returns:
            List of dictionaries containing:
                - 'bbox': (x, y, w, h)
                - 'emotion': Str (e.g. 'Happy')
                - 'confidence': Float (0.0 to 1.0)
                - 'probabilities': Dict[str, float]
                - 'face_crop': 48x48 numpy array
        """
        results = []
        if frame is None or frame.size == 0:
            return results

        if direct_crop:
            bboxes = [(0, 0, frame.shape[1], frame.shape[0])]
        else:
            bboxes = self.detector.detect_faces(frame)

        for i, bbox in enumerate(bboxes):
            try:
                face_crop, tensor = self.preprocessor.crop_and_preprocess(frame, bbox)
                tensor = tensor.to(self.device)

                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                # Temporal smoothing to stabilize video/webcam output
                if self.temporal_smoothing:
                    if i not in self._history:
                        self._history[i] = []
                    self._history[i].append(probs)
                    if len(self._history[i]) > 5:
                        self._history[i].pop(0)
                    smoothed_probs = np.mean(self._history[i], axis=0)
                else:
                    smoothed_probs = probs

                top_idx = int(np.argmax(smoothed_probs))
                top_emotion = EMOTION_LABELS.get(top_idx, "Unknown")
                top_conf = float(smoothed_probs[top_idx])

                prob_dict = {
                    EMOTION_LABELS[k]: float(smoothed_probs[k])
                    for k in range(len(EMOTION_LABELS))
                }

                results.append({
                    "bbox": bbox,
                    "emotion": top_emotion,
                    "confidence": top_conf,
                    "probabilities": prob_dict,
                    "face_crop": face_crop
                })
            except Exception as e:
                # Graceful skip for corrupted crops
                continue

        return results
