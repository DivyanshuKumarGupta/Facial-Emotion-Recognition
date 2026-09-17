"""
Evaluation Metrics & Visualization Utility Module.
Computes accuracy, precision, recall, F1-score, and renders confusion matrices.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config.config import EMOTION_LABELS, RESULTS_DIR


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes overall and per-class classification metrics.
    """
    if target_names is None:
        target_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
        
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    summary = {
        "accuracy": float(acc),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm
    }
    return summary


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Renders and optionally exports a formatted confusion matrix plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left")
    ax.set_yticklabels(class_names)

    # Label cells with counts
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.title("FER-2013 Confusion Matrix", pad=20, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()
