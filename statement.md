# Project Statement: Real-Time Facial Emotion Recognition

## 1. Problem Statement
Facial expressions are a key way people communicate emotions, but getting computers to recognize them accurately in real-time is challenging. Most simple computer vision programs struggle when tested outside controlled conditions because:

- **People look and express emotions differently**: Facial features vary widely across individuals, and expressions like fear and surprise can look almost identical around the eyes and mouth.
- **Lighting and webcam issues**: Uneven room lighting, harsh shadows, and low webcam resolutions reduce the quality of facial details needed for feature extraction.
- **Speed constraints**: To feel responsive in a live video stream, the system needs to detect faces and classify expressions within 25 to 35 milliseconds per frame without depending on expensive GPU hardware.

This project tackles these issues by building a practical emotion recognition pipeline. It combines OpenCV for face detection and illumination adjustment (CLAHE) with a custom PyTorch residual CNN trained on the FER-2013 dataset.

---

## 2. Scope of the Project
This project covers the full pipeline from raw camera input to emotion prediction:

- **Face Detection and Preprocessing**: Finding faces in images or video frames using Haar Cascades, adding margin padding around detected faces, applying CLAHE to balance contrast, and resizing each face crop to a standardized 48x48 grayscale image.
- **Neural Network Architecture**: Implementing a lightweight residual convolutional network (`EmotionResidualCNN`) in PyTorch with skip connections, batch normalization, and dropout to prevent overfitting while keeping the parameter count low (~1.5M parameters).
- **Classification into 7 Universal Classes**: Categorizing facial expressions into Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- **Command-Line Interface**: Providing CLI tools (`main.py`) to run predictions on single photos, batch-process video files, start live webcam inference, run training, and execute automated unit tests.
- **Benchmarking & Validation**: Evaluating performance using accuracy, class-level precision, recall, F1-scores, and confusion matrices.

### Out of Scope
- Detecting subtle micro-expressions that last less than 50 milliseconds.
- Audio or physiological signal analysis (this system relies purely on visual facial features).
- Identity verification or face recognition (only emotional expressions are classified).

---

## 3. Target Users
1. **Interactive Application Developers**: Programmers building games, interactive kiosks, or virtual assistants that adjust their responses based on how the user looks.
2. **Online Learning Platforms**: Educational software that tracks whether a student seems confused, bored, or engaged during recorded lectures.
3. **Usability & UX Researchers**: Teams testing digital products who want to see raw facial reactions to different user interface designs or video content.
4. **Computer Vision Students and Evaluators**: Academic reviewers looking for a clean, modular example of combining OpenCV preprocessing with deep learning classification.

---

## 4. High-Level Features
- **Live Webcam Mode**: Tracks faces and displays predicted emotions, confidence bars, and frame rates directly on the live video feed.
- **Headless CLI Execution**: Every major task (predicting images, processing videos, running tests, evaluating metrics) can be executed directly from the terminal without a GUI.
- **Contrast Normalization via CLAHE**: Preprocessing balances local lighting differences before images reach the neural network, making predictions more reliable under dim or uneven lighting.
- **Multi-Face Detection**: Detects and evaluates multiple faces within the same image or video frame.
- **Temporal Smoothing**: Uses a moving average across recent video frames to reduce rapid flickering between close emotion classes.
- **Detailed Metric Reports**: Generates confusion matrix heatmaps and JSON summaries showing precision and recall across all seven emotion classes.
- **Interactive Web UI**: Includes an optional Streamlit interface for uploading photos or testing the model through a web browser.
