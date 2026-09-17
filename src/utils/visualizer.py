"""
Visualizer Module for Computer Vision HUD & Bounding Box Annotations.
Renders stylized bounding boxes, emotion tags, probability meters, and FPS.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from config.config import EMOTION_COLORS


class Visualizer:
    """
    Renders high-clarity computer vision overlays onto frames.
    """
    def __init__(self, font_scale: float = 0.65, thickness: int = 2):
        self.font_scale = font_scale
        self.thickness = thickness
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_predictions(
        self,
        frame: np.ndarray,
        predictions: List[Dict[str, Any]],
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Draws bounding boxes and HUD stats on the frame.
        """
        annotated = frame.copy()
        
        # Display FPS indicator on top-left if provided
        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.rectangle(annotated, (15, 12), (135, 42), (20, 20, 20), -1)
            cv2.putText(annotated, fps_text, (22, 33), self.font, 0.6, (0, 255, 128), 2, cv2.LINE_AA)

        for pred in predictions:
            x, y, w, h = pred["bbox"]
            emotion = pred["emotion"]
            confidence = pred["confidence"]
            color = EMOTION_COLORS.get(emotion, (0, 255, 0))

            # Bounding box with corner accents
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, self.thickness)
            line_len = int(w * 0.2)
            # Top-left accent
            cv2.line(annotated, (x, y), (x + line_len, y), color, self.thickness + 2)
            cv2.line(annotated, (x, y), (x, y + line_len), color, self.thickness + 2)
            # Bottom-right accent
            cv2.line(annotated, (x + w, y + h), (x + w - line_len, y + h), color, self.thickness + 2)
            cv2.line(annotated, (x + w, y + h), (x + w, y + h - line_len), color, self.thickness + 2)

            # Emotion label header badge
            label_text = f"{emotion} ({confidence * 100:.1f}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, self.font, self.font_scale, 2)
            
            badge_y1 = max(0, y - text_h - 12)
            badge_y2 = y
            badge_x2 = min(frame.shape[1], x + text_w + 14)
            
            cv2.rectangle(annotated, (x, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(
                annotated,
                label_text,
                (x + 7, y - 6),
                self.font,
                self.font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # Confidence bar under bounding box
            bar_y = y + h + 8
            if bar_y + 8 < frame.shape[0]:
                cv2.rectangle(annotated, (x, bar_y), (x + w, bar_y + 6), (50, 50, 50), -1)
                cv2.rectangle(annotated, (x, bar_y), (x + int(w * confidence), bar_y + 6), color, -1)

        return annotated
