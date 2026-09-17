"""
Configuration Module for Facial Emotion Recognition System
Contains hyper-parameters, emotion class maps, file paths, and runtime settings.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SAMPLE_ASSETS_DIR = BASE_DIR / "sample_assets"
DOCS_DIR = BASE_DIR / "docs"

# Create standard directories if not present
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, SAMPLE_ASSETS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Emotion Classes Mapping (FER-2013 canonical standard)
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

EMOTION_COLORS = {
    "Angry": (0, 0, 230),        # Vivid Red
    "Disgust": (34, 139, 34),     # Forest Green
    "Fear": (148, 0, 211),       # Violet / Purple
    "Happy": (0, 215, 255),       # Gold / Amber
    "Sad": (255, 140, 0),         # Deep Sky Blue
    "Surprise": (255, 0, 255),    # Magenta
    "Neutral": (192, 192, 192)    # Silver Gray
}

# Image Preprocessing Hyperparameters
INPUT_SHAPE = (48, 48)            # FER-2013 native input size
NUM_CLASSES = 7
IN_CHANNELS = 1                   # Grayscale

# Training Default Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 20
RANDOM_SEED = 42

# Face Detection Settings
FACE_DETECTION_SCALE_FACTOR = 1.15
FACE_DETECTION_MIN_NEIGHBORS = 5
FACE_DETECTION_MIN_SIZE = (40, 40)

# Model Checkpoint Path
MODEL_WEIGHTS_PATH = MODELS_DIR / "emotion_model_weights.pth"
METRICS_OUTPUT_PATH = RESULTS_DIR / "evaluation_metrics.json"
