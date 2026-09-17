# Project Report: Real-Time Facial Emotion Recognition System 

---

## 1. Cover Page

- **Project Title**: Real-Time Facial Emotion Recognition System
- **Course**: Computer Vision
- **Student Name**: Divyanshu Kumar Gupta
- **Registration Number**: 24BAI10028
- **Institution**: VIT Bhopal University
- **Date**: 18 September 2026
- **Submission Type**: Build Your Own Project (BYOP) Capstone

---

## 2. Introduction

Facial expressions carry a significant amount of information during human interaction. Because of this, recognizing expressions automatically can be useful when building systems that need to respond to people in a more natural way.

In this project, I built a Computer Vision system that recognizes facial expressions from **webcam streams, recorded videos, and static images**. The model classifies a detected face into the seven emotion categories used by the FER-2013 dataset: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**.

The project combines traditional image-processing techniques with a deep learning model. OpenCV is used for face detection and preprocessing, while a custom Residual CNN is used for emotion classification.

---

## 3. Problem Statement

Facial emotion recognition becomes more difficult when the input comes from real-world environments instead of controlled datasets. Some of the main issues considered in this project are:

1. **Variation between faces and expressions**  
   People have different facial structures, and the same emotion can be expressed in slightly different ways. Some expressions can also look very similar, making classes such as *Fear* and *Surprise* harder to distinguish.

2. **Lighting and image quality**  
   Webcam and video inputs can contain shadows, uneven lighting, overexposure, or low-resolution faces. These changes can affect the visual features used by the classifier.

3. **Real-time processing requirements**  
   A model that is accurate but too slow is not very useful for live video. Large deep learning architectures can require considerable computational resources, so this project focuses on keeping the model relatively compact while still providing useful predictions.

To address these issues, the project uses a lightweight Residual CNN together with **Contrast Limited Adaptive Histogram Equalization (CLAHE)** during preprocessing.

---

## 4. Functional Requirements

The system is divided into five main modules.

### 1. Module 1 - Face Localization & Alignment

- Detect one or more faces from the input frame.
- Extract a bounding box for every detected face.
- Add padding around the detected region so that important parts of the face are less likely to be clipped.

### 2. Module 2 - Image Preprocessing & Illumination Equalization

- Apply **CLAHE** to improve local contrast.
- Convert detected faces into a `48 × 48` grayscale representation.
- Use random cropping, rotations, and horizontal flips during training as data augmentation.

### 3. Module 3 - Deep Residual Emotion Classification

- Pass the processed face through the Residual CNN.
- Predict one of the seven target emotion classes.
- Produce a Softmax probability distribution for the classes.

### 4. Module 4 - Visual HUD Rendering & Overlay

- Draw face bounding boxes.
- Display the predicted class and confidence.
- Show class probability information and the current FPS.

### 5. Module 5 - Multi-Mode CLI & Dashboard Interface

- Provide CLI commands for image prediction, video processing, training, evaluation, and testing.
- Provide a Streamlit dashboard for interactive experimentation.

---

## 5. Non-Functional Requirements

1. **Performance**: The system is designed for real-time processing, targeting more than 25-30 FPS on standard multi-core CPUs.
2. **Usability & Portability**: Core operations can be run from the terminal without depending on a graphical desktop environment.
3. **Reliability & Fault Tolerance**: The application handles cases such as frames with no detected faces and video streams ending earlier than expected.
4. **Maintainability & Modularity**: Configuration, model code, preprocessing, prediction, visualization, and utility functions are kept in separate modules. Automated tests are also included.

---

## 6. System Architecture

The system follows a pipeline-based design. Input is first passed through face detection and preprocessing, then through the deep learning model, and finally to the output layer.

```text
+------------------+     +-------------------+     +---------------------+
| Input Feed       | --> | Face Detection    | --> | Preprocessor        |
| (Webcam/Img/Vid) |     | (Haar Cascade)    |     | (CLAHE + 48x48 Norm)|
+------------------+     +-------------------+     +---------------------+
                                                              |
                                                              v
+------------------+     +-------------------+     +---------------------+
| Output Engine    | <-- | Softmax Classifier| <-- | EmotionResidualCNN  |
| (CLI/HUD/JSON)   |     | (7 Probabilities) |     | (Residual Blocks)   |
+------------------+     +-------------------+     +---------------------+
```

---

## 7. Design Diagrams

### 7.1 Use Case Diagram

The main actors and use cases for the system are:

- **Actors**: User / Evaluator, Video Camera / Image Source
- **Use Cases**:
  - `UC1`: Provide Image / Video via CLI
  - `UC2`: Launch Real-Time Webcam Stream
  - `UC3`: Train Model on FER-2013 Dataset
  - `UC4`: Evaluate Metrics & Confusion Matrix
  - `UC5`: Execute Automated Test Suite

### 7.2 Process Flow / Workflow Diagram

```text
Start -> Capture Frame -> Detect Faces
          |
          +--> [Faces Found?] -- No --> Return Clean Frame / Fallback Full ROI
          |
          +--> Yes
                |
                v
           Crop Face ROI -> Convert Grayscale -> CLAHE Equalization
                |
                v
           Resize to 48x48 -> Normalize [-1, 1] -> PyTorch Tensor
                |
                v
           Residual CNN Inference -> Softmax Class Probabilities
                |
                v
           Temporal Smoothing -> Annotate Bounding Box & HUD -> Display / Save
```

### 7.3 Sequence Diagram

```text
User -> Main CLI: python main.py predict --input image.jpg
Main CLI -> FaceDetector: detect_faces(image)
FaceDetector --> Main CLI: [bbox1, bbox2]
Main CLI -> Preprocessor: crop_and_preprocess(image, bbox1)
Preprocessor --> Main CLI: normalized_tensor (1, 1, 48, 48)
Main CLI -> EmotionResidualCNN: forward(tensor)
EmotionResidualCNN --> Main CLI: logits / probabilities
Main CLI -> Visualizer: draw_predictions(image, results)
Visualizer --> Main CLI: annotated_image
Main CLI -> Storage: save to results/
Main CLI --> User: Console Summary & JSON output
```

### 7.4 Class / Component Diagram

The main components are:

- `FaceDetector`: `detect_faces(frame: np.ndarray) -> List[Tuple]`
- `ImagePreprocessor`: `crop_and_preprocess(frame, bbox) -> (np.ndarray, Tensor)`
- `ResidualBlock(nn.Module)`: `forward(x: Tensor) -> Tensor`
- `EmotionResidualCNN(nn.Module)`: `forward(x), predict_probabilities(x)`
- `EmotionPredictor`: `predict_frame(frame, direct_crop) -> List[Dict]`
- `Visualizer`: `draw_predictions(frame, predictions, fps) -> np.ndarray`

---

## 8. Design Decisions & Rationale

### 1. PyTorch Framework Selection

PyTorch was selected because it provides a straightforward way to define the network, work with tensors, train the model, and modify the architecture when needed.

### 2. Residual Skip Connections

The model uses residual connections to help information and gradients pass through the network more effectively. This makes it possible to use multiple convolutional stages without making the network unnecessarily large. The model has a compact parameter footprint of approximately **1.5M parameters**.

### 3. Contrast Limited Adaptive Histogram Equalization (CLAHE)

Lighting can change considerably between different images and video frames. CLAHE is used to improve local contrast before classification while limiting excessive amplification of noise.

### 4. Grayscale 48x48 Resolution

The project uses grayscale `48 × 48` inputs to keep the amount of computation low. Facial expression information is largely represented through structures such as edges, shapes, and relative positions of facial regions, so color is not required by this particular input pipeline.

This also helps keep per-face processing lightweight, with the project targeting sub-15 ms inference.

### 5. CLI-First Architecture

The project was designed around a CLI so that the main operations can be executed without depending on a GUI. This also makes training, evaluation, testing, and batch processing easier to run from a terminal.

---

## 9. Implementation Details

The code is separated into packages according to their responsibilities:

- `config/config.py`: Stores configuration values, hyperparameters, color maps, and emotion labels.
- `src/models/emotion_cnn.py`: Defines the `EmotionResidualCNN` with three residual stages, dropout, and classification layers.
- `src/data/dataset.py`: Handles FER-2013 loading, data augmentation, and benchmark data preparation.
- `src/pipeline/face_detector.py`: Handles frontal face detection using the OpenCV Haar Cascade and applies padding to detected regions.
- `src/pipeline/preprocessor.py`: Performs face cropping, CLAHE enhancement, resizing, and normalization to the `[-1, 1]` range.
- `src/pipeline/emotion_predictor.py`: Connects detection, preprocessing, model inference, and temporal smoothing.
- `src/utils/visualizer.py`: Draws bounding boxes, probability information, and FPS on the output.
- `src/utils/metrics.py`: Calculates precision, recall, F1-scores, and confusion matrix visualizations.
- `main.py`: Provides the CLI commands `predict`, `video`, `live`, `train`, `evaluate`, and `test`.

---

## 10. Results

The current project results include:

- **Automated Test Results**: 8 unit tests passing for the model, tensor dimensions, preprocessing, and face detector.
- **Inference Speed**: Average inference time of **12.4 ms per face on CPU**, with video processing above **35 FPS**.
- **Classification Output**: Sample predictions include the predicted class, class probabilities, and bounding-box overlays saved under `results/`.
- **Confusion Matrix**: The generated confusion matrix is saved as `results/confusion_matrix.png`.

---

## 11. Testing Approach

Testing was done using `pytest` and covers both individual components and complete CLI flows.

### Unit Testing

The test suite checks:

- Tensor dimensions:

```text
(B, 1, 48, 48) -> (B, 7)
```

- Preprocessor output range:

```text
min >= -1.0
max <= 1.0
```

- Detector edge cases such as empty frames, `None` inputs, and synthetic test faces.

### Integration & Validation Testing

The CLI was also tested to make sure the main commands are routed correctly:

```text
predict
evaluate
train
test
```

---

## 12. Challenges Faced

### 1. Cross-Platform Face Detector Integration

One issue was making sure the Haar Cascade file could be located correctly across different operating systems. Using `cv2.data.haarcascades` avoided relying on hardcoded local paths.

### 2. Headless Execution Constraints

The project also needed to work in environments where a display or webcam might not be available. CLI-based operations were kept independent from the live display functionality, and fallback handling was added for these cases.

### 3. Illumination Disparities

Strong lighting differences sometimes affected the appearance of facial features. Applying CLAHE before normalization helped make the input more consistent before it reached the model.

---

## 13. Learnings & Key Takeaways

Working on this project provided practical experience in several areas:

- Designing and implementing a small residual CNN for a Computer Vision task.
- Combining classical Computer Vision methods such as Haar Cascades and CLAHE with a deep learning model.
- Building a complete inference pipeline instead of working only with the model itself.
- Organizing a project into separate modules for configuration, data, models, processing, visualization, and testing.
- Using automated tests to check individual components and CLI behavior.

---

## 14. Future Enhancements

There are several directions in which the system could be extended:

1. **Facial Landmark Attention**  
   Add 68-point facial landmark information so that the model can focus more on important regions such as the eyes and mouth.

2. **Temporal 3D-CNN / LSTM Modeling**  
   Instead of treating video frames mostly as individual inputs, temporal models could be used to learn how facial expressions change over a sequence of frames.

3. **Mobile & Edge Deployment**  
   The model could be optimized and quantized using technologies such as ONNX Runtime or INT8 TensorRT for deployment on embedded or edge devices.

---

## 15. References

1. Goodfellow, I. J., et al. (2013). *Challenges in representation learning: A report on three machine learning contests.* Neural Information Processing (ICONIP).
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
3. Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features.* CVPR.
4. OpenCV Library Documentation: https://docs.opencv.org/
5. PyTorch Deep Learning Platform: https://pytorch.org/
