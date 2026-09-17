"""
Streamlit Web Application: Real-Time Facial Emotion Recognition Dashboard.
Provides an interactive visual UI for uploading images/videos, inspecting emotion distributions,
and exploring evaluation metrics.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import MODEL_WEIGHTS_PATH, EMOTION_LABELS, RESULTS_DIR, SAMPLE_ASSETS_DIR
from src.pipeline.emotion_predictor import EmotionPredictor
from src.utils.visualizer import Visualizer

st.set_page_config(
    page_title="Facial Emotion Recognition AI",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    return EmotionPredictor(weights_path=MODEL_WEIGHTS_PATH)


predictor = load_predictor()
visualizer = Visualizer()

st.sidebar.title("🎭 FER-2013 System")
mode = st.sidebar.radio(
    "Navigation",
    ["Image Emotion Detection", "Model Performance & Metrics", "Sample Gallery", "About System"]
)

st.markdown('<div class="main-header">Facial Emotion Recognition System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Residual CNN trained on FER-2013 | Real-Time Computer Vision Pipeline</div>', unsafe_allow_html=True)

if mode == "Image Emotion Detection":
    st.subheader("Upload an Image for Analysis")
    uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1.2, 1])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    else:
        # Load default sample image
        sample_img_path = SAMPLE_ASSETS_DIR / "sample_face.jpg"
        if sample_img_path.exists():
            image = cv2.imread(str(sample_img_path))
            st.info("Displaying default test sample face. Upload your own image above to test!")
        else:
            image = None

    if image is not None:
        predictions = predictor.predict_frame(image)
        if not predictions:
            predictions = predictor.predict_frame(image, direct_crop=True)

        annotated = visualizer.draw_predictions(image, predictions)
        rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        with col1:
            st.image(rgb_annotated, caption="Annotated Detections", use_container_width=True)

        with col2:
            st.markdown("### Detection Insights")
            if predictions:
                for idx, pred in enumerate(predictions):
                    st.markdown(f"**Face #{idx + 1}: Predicted Emotion: `{pred['emotion']}`**")
                    st.progress(pred["confidence"])
                    st.write(f"Confidence: **{pred['confidence'] * 100:.1f}%**")

                    df_probs = pd.DataFrame(
                        list(pred["probabilities"].items()),
                        columns=["Emotion", "Probability"]
                    )
                    fig = px.bar(
                        df_probs,
                        x="Emotion",
                        y="Probability",
                        color="Emotion",
                        range_y=[0, 1],
                        title=f"Face #{idx+1} Emotion Distribution"
                    )
                    fig.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No face detected in the given image.")

elif mode == "Model Performance & Metrics":
    st.subheader("Model Evaluation & Benchmark Report")
    
    metrics_file = RESULTS_DIR / "evaluation_metrics.json"
    cm_file = RESULTS_DIR / "confusion_matrix.png"
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Architecture", "ResNet-FER (CNN)")
    col2.metric("Input Dimension", "48 x 48 (1-ch)")
    col3.metric("Emotion Classes", "7 Universal")

    if cm_file.exists():
        st.markdown("### Confusion Matrix Heatmap")
        st.image(str(cm_file), caption="FER-2013 Benchmark Confusion Matrix", use_container_width=False, width=650)
    else:
        st.info("Run `python main.py evaluate` to generate evaluation metrics and confusion matrix plot.")

elif mode == "Sample Gallery":
    st.subheader("Test Asset Gallery")
    samples = list(SAMPLE_ASSETS_DIR.glob("*.jpg"))
    if samples:
        cols = st.columns(len(samples))
        for i, s_path in enumerate(samples):
            with cols[i]:
                s_img = Image.open(s_path)
                st.image(s_img, caption=s_path.name, use_container_width=True)
    else:
        st.write("No sample images found.")

elif mode == "About System":
    st.subheader("System Architecture & Specifications")
    st.markdown("""
    - **Dataset**: FER-2013 (Facial Expression Recognition 2013)
    - **Backbone Network**: `EmotionResidualCNN` featuring 3 residual feature extraction stages with batch normalization, dropout, and residual skip connections.
    - **Face Detection**: OpenCV CascadeClassifier with margin padding and CLAHE illumination enhancement.
    - **CLI Interface**: Strict CLI execution available via `python main.py predict`, `python main.py video`, `python main.py evaluate`, and `python main.py test`.
    """)
