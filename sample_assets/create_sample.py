"""
Sample Asset Generator & Model Weights Initializer.
Creates test images in sample_assets/ and initializes baseline model weights.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import SAMPLE_ASSETS_DIR, MODEL_WEIGHTS_PATH, MODELS_DIR
from src.models.emotion_cnn import EmotionResidualCNN
from src.data.dataset import create_benchmark_split, get_data_loaders


def generate_test_face(emotion: str = "Happy", save_path: Path = None) -> np.ndarray:
    """
    Synthesizes a clean 300x300 test face image for testing CLI and vision pipelines.
    """
    img = np.full((300, 300, 3), 235, dtype=np.uint8)
    center = (150, 150)
    
    # Face oval
    cv2.ellipse(img, center, (90, 115), 0, 0, 360, (190, 205, 220), -1)
    cv2.ellipse(img, center, (90, 115), 0, 0, 360, (90, 95, 105), 3)
    
    # Eyes
    cv2.circle(img, (115, 125), 14, (255, 255, 255), -1)
    cv2.circle(img, (185, 125), 14, (255, 255, 255), -1)
    cv2.circle(img, (115, 125), 6, (40, 40, 40), -1)
    cv2.circle(img, (185, 125), 6, (40, 40, 40), -1)
    
    # Eyebrows
    if emotion == "Angry":
        cv2.line(img, (95, 100), (130, 115), (30, 30, 30), 4)
        cv2.line(img, (205, 100), (170, 115), (30, 30, 30), 4)
    elif emotion == "Surprise":
        cv2.ellipse(img, (115, 100), (18, 10), 0, 180, 360, (30, 30, 30), 3)
        cv2.ellipse(img, (185, 100), (18, 10), 0, 180, 360, (30, 30, 30), 3)
    else:
        cv2.line(img, (100, 108), (130, 108), (30, 30, 30), 3)
        cv2.line(img, (170, 108), (200, 108), (30, 30, 30), 3)

    # Nose
    cv2.line(img, (150, 135), (145, 160), (90, 95, 105), 2)
    cv2.line(img, (145, 160), (155, 160), (90, 95, 105), 2)

    # Mouth expressions
    if emotion == "Happy":
        cv2.ellipse(img, (150, 180), (38, 22), 0, 0, 180, (40, 40, 180), -1)
        cv2.ellipse(img, (150, 180), (38, 22), 0, 0, 180, (20, 20, 120), 2)
    elif emotion == "Sad":
        cv2.ellipse(img, (150, 205), (35, 18), 0, 180, 360, (40, 40, 160), 3)
    elif emotion == "Angry":
        cv2.line(img, (125, 195), (175, 195), (40, 40, 150), 4)
    elif emotion == "Surprise":
        cv2.circle(img, (150, 195), 18, (40, 40, 150), -1)
    else: # Neutral
        cv2.line(img, (130, 195), (170, 195), (50, 50, 50), 3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), img)
        
    return img


def initialize_model_weights():
    """
    Initializes and calibrates weights for EmotionResidualCNN.
    Trains on benchmark split for 5 quick epochs so weights are valid and functional.
    """
    print("[*] Initializing model weights on benchmark split...")
    train_ds, val_ds = create_benchmark_split(num_samples_per_class=50, seed=42)
    train_loader, _ = get_data_loaders(train_ds, val_ds, batch_size=32)
    
    device = torch.device("cpu")
    model = EmotionResidualCNN(in_channels=1, num_classes=7)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    for epoch in range(5):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"[+] Model weights saved to {MODEL_WEIGHTS_PATH}")


if __name__ == "__main__":
    generate_test_face("Happy", SAMPLE_ASSETS_DIR / "sample_face.jpg")
    generate_test_face("Angry", SAMPLE_ASSETS_DIR / "angry_face.jpg")
    generate_test_face("Surprise", SAMPLE_ASSETS_DIR / "surprise_face.jpg")
    print(f"[+] Generated sample test face images in {SAMPLE_ASSETS_DIR}")
    initialize_model_weights()
