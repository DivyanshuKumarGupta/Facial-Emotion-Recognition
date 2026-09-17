"""
Main Command-Line Interface (CLI) for Real-Time Facial Emotion Recognition System.
Provides full terminal-driven executability for headless environments and local execution.

Usage:
    python main.py predict --input sample_assets/sample_face.jpg
    python main.py evaluate
    python main.py train --epochs 5
    python main.py test
    python main.py video --input test_video.mp4 --output results/output.mp4
    python main.py live
"""

import sys
import os
import argparse
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import (
    MODEL_WEIGHTS_PATH,
    RESULTS_DIR,
    SAMPLE_ASSETS_DIR,
    EMOTION_LABELS,
    METRICS_OUTPUT_PATH,
    BATCH_SIZE,
    LEARNING_RATE
)
from src.pipeline.emotion_predictor import EmotionPredictor
from src.utils.visualizer import Visualizer
from src.utils.metrics import calculate_metrics, plot_confusion_matrix
from src.data.dataset import create_benchmark_split, get_data_loaders, load_fer2013_from_csv
from src.models.emotion_cnn import EmotionResidualCNN


def command_predict(args):
    """Executes single image prediction in headless or visual mode."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[-] Error: Input file '{input_path}' does not exist.")
        sys.exit(1)

    print(f"[*] Processing image: {input_path}")
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"[-] Error: Unable to decode image at '{input_path}'.")
        sys.exit(1)

    predictor = EmotionPredictor(weights_path=MODEL_WEIGHTS_PATH)
    predictions = predictor.predict_frame(image)

    if not predictions:
        print("[!] No face detected with default cascade. Retrying with full-image ROI...")
        predictions = predictor.predict_frame(image, direct_crop=True)

    print("\n================ PREDICTION RESULTS ================")
    for i, pred in enumerate(predictions):
        bbox = pred["bbox"]
        emotion = pred["emotion"]
        conf = pred["confidence"]
        print(f"\n[Face {i+1}]")
        print(f"  Bounding Box  : (x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]})")
        print(f"  Top Emotion   : {emotion.upper()}")
        print(f"  Confidence    : {conf * 100:.2f}%")
        print("  Class Probabilities:")
        for emo, p in pred["probabilities"].items():
            bar = "#" * int(p * 25)
            print(f"    - {emo:<10} : {p*100:5.2f}% | {bar}")

    # Annotate and save output
    visualizer = Visualizer()
    annotated = visualizer.draw_predictions(image, predictions)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"\n[+] Annotated result saved to: {output_path}")

    if args.json:
        # Output clean JSON for automation pipelines
        clean_preds = [
            {
                "bbox": p["bbox"],
                "emotion": p["emotion"],
                "confidence": p["confidence"],
                "probabilities": p["probabilities"]
            }
            for p in predictions
        ]
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(clean_preds, indent=2))


def command_video(args):
    """Processes video file frame-by-frame and exports annotated video."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[-] Error: Video file '{input_path}' not found.")
        sys.exit(1)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[-] Error: Could not open video file '{input_path}'.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    predictor = EmotionPredictor(weights_path=MODEL_WEIGHTS_PATH)
    visualizer = Visualizer()

    print(f"[*] Processing video ({total_frames} frames, {fps:.1f} FPS)...")
    frame_idx = 0
    t_start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        preds = predictor.predict_frame(frame)
        current_fps = frame_idx / (time.time() - t_start + 1e-5)
        annotated = visualizer.draw_predictions(frame, preds, fps=current_fps)
        writer.write(annotated)

        if frame_idx % 25 == 0 or frame_idx == total_frames:
            print(f"  -> Frame {frame_idx}/{total_frames} ({current_fps:.1f} FPS)")

    cap.release()
    writer.release()
    print(f"[+] Annotated video saved to: {output_path}")


def command_live(args):
    """Real-time webcam inference stream."""
    print("[*] Initializing webcam video feed (Index: {})...".format(args.camera))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[-] Error: Could not open camera device.")
        print("[!] Note: If running in a headless or remote shell without physical camera,")
        print("    please use 'python main.py predict --input sample_assets/sample_face.jpg' instead.")
        return

    predictor = EmotionPredictor(weights_path=MODEL_WEIGHTS_PATH)
    visualizer = Visualizer()

    prev_time = time.time()
    print("[+] Camera stream active! Press 'q' or 'ESC' in the display window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[-] End of stream or camera disconnected.")
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time

        preds = predictor.predict_frame(frame)
        annotated = visualizer.draw_predictions(frame, preds, fps=fps)

        try:
            cv2.imshow("Real-Time Facial Emotion Recognition", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        except Exception:
            print("[!] Headless environment detected: Unable to open OpenCV GUI window.")
            print("[!] Running single frame headless benchmark instead.")
            break

    cap.release()
    cv2.destroyAllWindows()


def command_train(args):
    """Trains or fine-tunes EmotionResidualCNN."""
    print("[*] Setting up training environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device}")

    if args.data_csv and Path(args.data_csv).exists():
        print(f"[*] Loading FER-2013 data from CSV: {args.data_csv}")
        train_ds = load_fer2013_from_csv(Path(args.data_csv), usage_filter="Training")
        val_ds = load_fer2013_from_csv(Path(args.data_csv), usage_filter="PublicTest")
    else:
        print("[*] No custom CSV specified. Training on FER-2013 Benchmark Data Split...")
        train_ds, val_ds = create_benchmark_split(num_samples_per_class=60, seed=42)

    train_loader, val_loader = get_data_loaders(train_ds, val_ds, batch_size=args.batch_size)

    model = EmotionResidualCNN(in_channels=1, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    epochs = args.epochs
    print(f"[*] Starting training loop for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (train_correct / total_train) * 100.0

        # Validation phase
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (val_correct / total_val) * 100.0
        scheduler.step(epoch_val_loss)

        print(f"  Epoch [{epoch:2d}/{epochs:2d}] | "
              f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:5.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:5.2f}%")

    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    print(f"\n[+] Successfully saved trained model weights to: {MODEL_WEIGHTS_PATH}")


def command_evaluate(args):
    """Evaluates the model and computes precision, recall, F1, and confusion matrix."""
    print("[*] Evaluating EmotionResidualCNN performance...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.data_csv and Path(args.data_csv).exists():
        val_ds = load_fer2013_from_csv(Path(args.data_csv), usage_filter="PrivateTest")
    else:
        _, val_ds = create_benchmark_split(num_samples_per_class=40, seed=123)

    _, val_loader = get_data_loaders(val_ds, val_ds, batch_size=32)
    predictor = EmotionPredictor(weights_path=MODEL_WEIGHTS_PATH, device=device)
    model = predictor.model

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.tolist())

    metrics = calculate_metrics(y_true, y_pred)
    print("\n================ EVALUATION METRICS ================")
    print(f"Overall Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score   : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print("\nPer-Class Breakdown:")
    rep = metrics["classification_report"]
    for emo_id, emo_name in EMOTION_LABELS.items():
        c_metrics = rep.get(emo_name, {})
        p = c_metrics.get("precision", 0.0)
        r = c_metrics.get("recall", 0.0)
        f1 = c_metrics.get("f1-score", 0.0)
        sup = c_metrics.get("support", 0)
        print(f"  - {emo_name:<10}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}, Support={sup}")

    # Save JSON metrics
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] Saved metrics JSON to: {METRICS_OUTPUT_PATH}")

    # Plot & save Confusion Matrix
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    class_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), class_names, save_path=cm_path)
    print(f"[+] Saved confusion matrix heatmap to: {cm_path}")


def command_test(args):
    """Runs automated test suite."""
    import subprocess
    print("[*] Running automated test suite with pytest...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"])
    sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Facial Emotion Recognition System (FER-2013)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="System command to execute")

    # Predict command
    p_pred = subparsers.add_parser("predict", help="Predict emotion on a single image")
    p_pred.add_argument("--input", "-i", type=str, default=str(SAMPLE_ASSETS_DIR / "sample_face.jpg"), help="Path to input image")
    p_pred.add_argument("--output", "-o", type=str, default=str(RESULTS_DIR / "prediction_result.jpg"), help="Path to output annotated image")
    p_pred.add_argument("--json", action="store_true", help="Print structured JSON output")

    # Video command
    p_vid = subparsers.add_parser("video", help="Process a video file")
    p_vid.add_argument("--input", "-i", type=str, required=True, help="Input video file path")
    p_vid.add_argument("--output", "-o", type=str, default=str(RESULTS_DIR / "annotated_video.mp4"), help="Output video file path")
    p_vid.add_argument("--max-frames", type=int, default=None, help="Limit maximum frames to process")

    # Live command
    p_live = subparsers.add_parser("live", help="Launch live webcam inference")
    p_live.add_argument("--camera", "-c", type=int, default=0, help="Camera index")

    # Train command
    p_train = subparsers.add_parser("train", help="Train EmotionResidualCNN model")
    p_train.add_argument("--epochs", "-e", type=int, default=5, help="Number of training epochs")
    p_train.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    p_train.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    p_train.add_argument("--data-csv", type=str, default=None, help="Optional path to fer2013.csv")

    # Evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Compute metrics and confusion matrix")
    p_eval.add_argument("--data-csv", type=str, default=None, help="Optional path to fer2013.csv")

    # Test command
    p_test = subparsers.add_parser("test", help="Run unit test suite")

    args = parser.parse_args()

    if args.command == "predict":
        command_predict(args)
    elif args.command == "video":
        command_video(args)
    elif args.command == "live":
        command_live(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "evaluate":
        command_evaluate(args)
    elif args.command == "test":
        command_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
