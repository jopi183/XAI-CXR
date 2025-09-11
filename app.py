import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import plotly.express as px
import pandas as pd
import os 
from captum.attr import Saliency
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import EfficientNetClassifier

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Image Classifier with XAI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    
    .header-container {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 3rem 2rem; 
        border-radius: 8px; 
        margin-bottom: 3rem;
        text-align: center; 
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 2.5rem; 
        font-weight: 300; 
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        font-size: 1.1rem; 
        opacity: 0.8; 
        margin-bottom: 0;
        font-weight: 300;
    }
    
    .content-card {
        background: white;
        padding: 2rem; 
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 2rem; 
        border: 1px solid #e8ecef;
    }
    
    .upload-area {
        border: 2px dashed #bdc3c7; 
        border-radius: 8px;
        padding: 3rem; 
        text-align: center;
        background: #fbfcfd;
        margin: 1.5rem 0; 
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #95a5a6;
        background: #f8f9fa;
    }
    
    .results-section {
        background: white; 
        padding: 2.5rem; 
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06); 
        margin-top: 2rem;
        border: 1px solid #e8ecef;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #34495e, #2c3e50);
        padding: 2rem; 
        border-radius: 8px;
        color: white; 
        text-align: center; 
        margin-bottom: 2rem;
    }
    
    .prediction-class {
        font-size: 1.8rem; 
        font-weight: 300; 
        margin-bottom: 0.5rem;
        letter-spacing: -0.3px;
    }
    
    .prediction-confidence {
        font-size: 1.1rem; 
        opacity: 0.9;
        font-weight: 300;
    }
    
    .xai-section-title {
        font-size: 1.5rem;
        font-weight: 300;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: -0.3px;
    }
    
    .xai-method-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border: 1px solid #e8ecef;
    }
    
    .xai-visualization-card {
        background: white; 
        padding: 1.5rem; 
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem; 
        border: 1px solid #e8ecef;
    }
    
    .xai-card-title {
        color: #2c3e50; 
        font-size: 1.1rem; 
        font-weight: 400;
        margin-bottom: 1rem; 
        text-align: center;
        letter-spacing: -0.2px;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 400;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        letter-spacing: -0.2px;
    }
    
    .stRadio > div {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        justify-content: center;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border: 1px solid #bdc3c7;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 400;
        color: #2c3e50;
    }
    
    .stRadio > div > label:hover {
        border-color: #95a5a6;
        background: #f8f9fa;
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: #34495e;
        color: white;
        border-color: #34495e;
    }
    
    .stButton > button {
        border-radius: 6px !important;
        font-weight: 400 !important;
        transition: all 0.3s ease !important;
        border: 1px solid #bdc3c7 !important;
        box-shadow: none !important;
        letter-spacing: -0.1px !important;
    }
    
    .stButton > button:hover {
        border-color: #95a5a6 !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: #34495e !important;
        border-color: #34495e !important;
        color: white !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: white !important;
        color: #2c3e50 !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    h1, h2, h3, h4, h5, h6 {
        font-weight: 300 !important;
        letter-spacing: -0.3px !important;
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model(model_path, device):
    try:
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan: {model_path}")
            return None, None
        
        checkpoint = torch.load(model_path, map_location=device)
        class_names = checkpoint.get('class_names')
        num_classes = checkpoint.get('num_classes')

        if num_classes is None:
            st.error("Checkpoint tidak berisi 'num_classes'.")
            return None, None

        model = EfficientNetClassifier(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, class_names
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

class ImageProcessor:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        ])
    
    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            image = Image.fromarray(image).convert('RGB')
        
        original_image = np.array(image)
        input_tensor = self.transform(image).unsqueeze(0)
        
        return input_tensor, original_image
    
    def denormalize_tensor(self, tensor):
        return self.denormalize(tensor)

class ModelLoader:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names = load_cached_model(model_path, self.device)

    def is_ready(self):
        return self.model is not None and self.class_names is not None

    def predict(self, input_tensor):
        if not self.is_ready():
            raise ValueError("Model belum dimuat atau gagal dimuat!")
        
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        return predicted_class, probabilities.cpu().numpy()[0]

class XAIVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.saliency = Saliency(model)
        
        self.target_layers = [model.efficientnet.features[-1]]
        self.grad_cam = GradCAM(model=model, target_layers=self.target_layers)
        self.score_cam = ScoreCAM(model=model, target_layers=self.target_layers)
    
    def generate_saliency_map(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        
        try:
            attribution = self.saliency.attribute(input_tensor, target=target_class)
            return attribution.squeeze().cpu().detach().numpy()
        except Exception as e:
            st.error(f"Error dalam generate_saliency_map: {str(e)}")
            return None
    
    def generate_grad_cam(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        
        try:
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = self.grad_cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0, :]
        except Exception as e:
            st.error(f"Error dalam generate_grad_cam: {str(e)}")
            return None
    
    def generate_score_cam(self, input_tensor, target_class):
        try:
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            targets = [ClassifierOutputTarget(target_class)]
            
            score_cam = ScoreCAM(model=self.model, target_layers=[self.model.efficientnet.features[-1]])
            
            grayscale_cam = score_cam(input_tensor=input_tensor, targets=targets)
            
            heatmap = grayscale_cam[0, :]
            heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
            return heatmap
        
        except Exception as e:
            st.error(f"Error dalam generate_score_cam: {str(e)}")
            return None
    
    def create_heatmap_overlay(self, original_image, heatmap, alpha=0.4):
        if heatmap is None:
            return original_image
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        original_normalized = original_image.astype(np.float32) / 255.0
        blended = (1 - alpha) * original_normalized + alpha * heatmap_colored
        
        return (blended * 255).astype(np.uint8)

def create_prediction_chart(probabilities, class_names):
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': probabilities * 100
    })
    df = df.sort_values('Probability', ascending=True)

    fig = px.bar(
        df, 
        x='Probability', 
        y='Class',
        orientation='h',
        color='Probability',
        color_continuous_scale=['#ecf0f1', '#34495e'],
        title='Classification Probabilities'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        title_x=0.5,
        title_font_family="Arial",
        title_font_color="#2c3e50",
        xaxis_title="Probability (%)",
        yaxis_title="Class",
        coloraxis_showscale=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_color="#2c3e50"
    )
    
    return fig

def display_xai_visualization(original_image, attribution, title, method_type='heatmap'):
    if attribution is None:
        st.warning(f"Tidak dapat menghasilkan visualisasi untuk {title}")
        return
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="xai-visualization-card">
            <div class="xai-card-title">Original Image</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="xai-visualization-card">
            <div class="xai-card-title">{title}</div>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            if method_type == 'heatmap':
                visualizer = st.session_state.xai_visualizer
                overlay = visualizer.create_heatmap_overlay(original_image, attribution)
                st.image(overlay, caption=f"{title} Overlay", use_container_width=True)
            else:
                attribution_normalized = np.transpose(attribution, (1, 2, 0))
                attribution_gray = np.mean(np.abs(attribution_normalized), axis=2)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(attribution_gray, cmap='hot', alpha=0.8)
                ax.imshow(original_image, alpha=0.3)
                ax.axis('off')
                ax.set_title(f'{title}', fontsize=14, fontweight='300', color='#2c3e50')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close()
        except Exception as e:
            st.error(f"Error dalam visualisasi {title}: {str(e)}")

def main():
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Chest X-ray Classification</div>
        <div class="header-subtitle">AI-powered medical image analysis with explainable AI</div>
    </div>
    """, unsafe_allow_html=True)
    
    from pathlib import Path

    APP_DIR = Path(__file__).resolve().parent
    MODEL_PATH = APP_DIR / "efficientnet_b0_classifier.pth"

    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor()
    
    if 'model_loader' not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.model_loader = ModelLoader(MODEL_PATH)
            st.success("Model loaded successfully")

    if st.session_state.model_loader.is_ready():
        if 'xai_visualizer' not in st.session_state:
            try:
                with st.spinner("Initializing XAI visualizer..."):
                    st.session_state.xai_visualizer = XAIVisualizer(
                        st.session_state.model_loader.model,
                        st.session_state.model_loader.device
                    )
                    st.success("XAI visualizer initialized successfully")
            except Exception as e:
                st.error(f"Failed to initialize XAI: {e}")
                st.stop()

        st.markdown("""
        <div class="content-card">
            <h3 class="section-title">Upload Chest X-ray Image</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing image..."):
                    original_image = Image.open(uploaded_file)
                    input_tensor, original_array = st.session_state.image_processor.preprocess_image(original_image)
                
                with st.spinner("Making prediction..."):
                    predicted_class, probabilities = st.session_state.model_loader.predict(input_tensor)
                    predicted_class_name = st.session_state.model_loader.class_names[predicted_class]
                    confidence = probabilities[predicted_class] * 100
                
                st.markdown("""
                <div class="results-section">
                    <h2 class="section-title">Classification Results</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="prediction-container">
                    <div class="prediction-class">Prediction: {predicted_class_name}</div>
                    <div class="prediction-confidence">Confidence: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(original_image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    fig = create_prediction_chart(probabilities, st.session_state.model_loader.class_names)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                <div class="results-section">
                    <h2 class="xai-section-title">Explainable AI Analysis</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="xai-method-container">
                    <h4 class="section-title">Select Visualization Method</h4>
                </div>
                """, unsafe_allow_html=True)
                
                xai_options = [
                    "Saliency Map",
                    "Grad-CAM",
                    "Score-CAM"
                ]
                
                selected_method = st.radio(
                    "Visualization Method:",
                    xai_options,
                    index=0
                )
                
                with st.spinner(f"Generating {selected_method} visualization..."):
                    if selected_method == "Saliency Map":
                        attribution = st.session_state.xai_visualizer.generate_saliency_map(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Saliency Map", method_type='gradient'
                        )
                    
                    elif selected_method == "Grad-CAM":
                        attribution = st.session_state.xai_visualizer.generate_grad_cam(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Grad-CAM", method_type='heatmap'
                        )
                    
                    elif selected_method == "Score-CAM":
                        attribution = st.session_state.xai_visualizer.generate_score_cam(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Score-CAM", method_type='heatmap'
                        )
                
                st.success(f"{selected_method} visualization generated successfully")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error("Please try again with a different image.")
    else:
        st.error(f"Failed to load model. Please ensure '{MODEL_PATH}' exists and is not corrupted.")
        st.info("Make sure the model file 'efficientnet_b0_classifier.pth' is available in the application directory.")
        st.stop()

if __name__ == "__main__":
    main()