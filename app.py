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
from captum.attr import Saliency, IntegratedGradients
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import EfficientNetClassifier

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Image Classifier with XAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main > div { padding-top: 2rem; }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
        text-align: center; color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        font-size: 1.2rem; opacity: 0.9; margin-bottom: 0;
    }
    
    .info-card {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        padding: 1.5rem; border-radius: 15px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1), -5px -5px 15px rgba(255,255,255,0.7);
        margin-bottom: 1.5rem; border: 1px solid rgba(255,255,255,0.2);
    }
    
    .upload-area {
        border: 3px dashed #667eea; border-radius: 15px;
        padding: 2rem; text-align: center;
        background: linear-gradient(145deg, #f8f9ff, #ffffff);
        margin: 1rem 0; transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #764ba2;
        background: linear-gradient(145deg, #ffffff, #f8f9ff);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    
    .results-container {
        background: white; padding: 2rem; border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1); margin-top: 2rem;
    }
    
    .xai-card {
        background: white; padding: 1.5rem; border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem; border: 1px solid #e1e8ed;
        transition: transform 0.2s ease;
    }
    
    .xai-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .xai-title {
        color: #2c3e50; font-size: 1.1rem; font-weight: 600;
        margin-bottom: 0.5rem; text-align: center;
    }
    
    
    .prediction-container {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1.5rem; border-radius: 12px;
        color: white; text-align: center; margin-bottom: 1.5rem;
    }
    
    .prediction-class {
        font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;
    }
    
    .prediction-confidence {
        font-size: 1rem; opacity: 0.9;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        # Panggil fungsi cache saat inisialisasi
        self.model, self.class_names = load_cached_model(model_path, self.device)

    def is_ready(self):
        """Memeriksa apakah model berhasil dimuat."""
        return self.model is not None

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
        self.integrated_gradients = IntegratedGradients(model)
        
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
    
    def generate_integrated_gradients(self, input_tensor, target_class, steps=50):
        input_tensor = input_tensor.to(self.device)
        
        try:
            attribution = self.integrated_gradients.attribute(
                input_tensor, 
                target=target_class, 
                n_steps=steps
            )
            return attribution.squeeze().cpu().detach().numpy()
        except Exception as e:
            st.error(f"Error dalam generate_integrated_gradients: {str(e)}")
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
            
            score_cam = ScoreCAM(model=self.model, target_layers=[self.model.efficientnet.features[-2]])
            
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
        color_continuous_scale='Viridis',
        title='Prediksi Probabilitas Kelas'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        title_x=0.5,
        xaxis_title="Probabilitas (%)",
        yaxis_title="Kelas",
        coloraxis_showscale=False
    )
    
    return fig

def display_xai_visualization(original_image, attribution, title, method_type='heatmap'):
    if attribution is None:
        st.warning(f"⚠️ Tidak dapat menghasilkan visualisasi untuk {title}")
        return
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="xai-card">
            <div class="xai-title">{title}</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(original_image, caption="Gambar Asli", use_container_width=True)
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        try:
            if method_type == 'heatmap':
                visualizer = st.session_state.xai_visualizer
                overlay = visualizer.create_heatmap_overlay(original_image, attribution)
                st.image(overlay, caption=f"{title} Visualization", use_container_width=True)
            else:
                attribution_normalized = np.transpose(attribution, (1, 2, 0))
                attribution_gray = np.mean(np.abs(attribution_normalized), axis=2)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(attribution_gray, cmap='hot', alpha=0.8)
                ax.imshow(original_image, alpha=0.3)
                ax.axis('off')
                ax.set_title(f'{title} Visualization', fontsize=14, fontweight='bold')
                
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
        <div class="header-title">🔍 Chest X-ray Classification</div>
        <div class="header-subtitle">with Explainable AI (XAI) Visualization</div>
    </div>
    """, unsafe_allow_html=True)
    
    from pathlib import Path

    APP_DIR = Path(__file__).resolve().parent
    MODEL_PATH = APP_DIR / "efficientnet_b0_classifier.pth"

    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor()

    if not st.session_state.model_loaded:
        with st.spinner(f"Memuat model '{MODEL_PATH}', harap tunggu..."):
            success, message = st.session_state.model_loader.load_model(MODEL_PATH)
        
        if success:
            st.session_state.model_loaded = True
            try:
                st.session_state.xai_visualizer = XAIVisualizer(
                    st.session_state.model_loader.model,
                    st.session_state.model_loader.device
                )
                st.success(message)
            except Exception as xai_error:
                st.error(f"❌ Failed to initialize XAI visualizer: {str(xai_error)}")
                st.session_state.model_loaded = False
        else:
            st.error(f"Gagal memuat model. Pastikan file '{MODEL_PATH}' ada di direktori yang sama dengan skrip ini. Detail Error: {message}")
    
    if st.session_state.model_loaded:
        st.markdown("""
        <div class="info-card">
            <h3>📸 Upload Gambar Chest X-ray</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Pilih gambar...",
            type=['png', 'jpg', 'jpeg'],
            help="Format yang didukung: PNG, JPG, JPEG"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            
            with st.spinner("Memproses gambar..."):
                input_tensor, original_array = st.session_state.image_processor.preprocess_image(original_image)
            
            with st.spinner("Melakukan prediksi..."):
                predicted_class, probabilities = st.session_state.model_loader.predict(input_tensor)
                predicted_class_name = st.session_state.model_loader.class_names[predicted_class]
                confidence = probabilities[predicted_class] * 100
            
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(original_array, caption="Gambar yang Diupload", use_container_width=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-container">
                    <div class="prediction-class">Prediksi: {predicted_class_name}</div>
                    <div class="prediction-confidence">Confidence: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                fig = create_prediction_chart(probabilities, st.session_state.model_loader.class_names)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("## 🧠 Explainable AI (XAI) Visualizations")
            st.markdown("Berikut adalah empat metode XAI yang menjelaskan bagaimana model membuat keputusan:")
            
            with st.spinner("Generating XAI visualizations..."):
                
                st.markdown("### 1. Saliency Maps")
                saliency_attr = st.session_state.xai_visualizer.generate_saliency_map(
                    input_tensor, predicted_class
                )
                display_xai_visualization(
                    original_array,
                    saliency_attr,
                    "Saliency Maps",
                    method_type='gradient'
                )
                
                st.markdown("---")
                
                st.markdown("### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)")
                grad_cam_result = st.session_state.xai_visualizer.generate_grad_cam(
                    input_tensor, predicted_class
                )
                display_xai_visualization(
                    original_array,
                    grad_cam_result,
                    "Grad-CAM",
                    method_type='heatmap'
                )
                
                st.markdown("---")
                
                st.markdown("### 3. Integrated Gradients")
                ig_attr = st.session_state.xai_visualizer.generate_integrated_gradients(
                    input_tensor, predicted_class
                )
                display_xai_visualization(
                    original_array,
                    ig_attr,
                    "Integrated Gradients",
                    method_type='gradient'
                )
                
                st.markdown("---")
                
                st.markdown("### 4. Score-CAM")
                score_cam_result = st.session_state.xai_visualizer.generate_score_cam(
                    input_tensor, predicted_class
                )
                display_xai_visualization(
                    original_array,
                    score_cam_result,
                    "Score-CAM",
                    method_type='heatmap'
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 20px; margin: 2rem 0;
                        color: white; text-align: center;">
                <h3 style="margin-bottom: 1rem; font-weight: 700;">Klasifikasi Chest X-ray dengan XAI</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0; opacity: 0.9;">
                    Membandingkan berbagai metode XAI di atas untuk memahami bagaimana model Anda membuat keputusan. 
                    Setiap metode memberikan wawasan yang unik mengenai proses penalaran AI.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p><em>Explainable AI untuk Klasifikasi Gambar yang Transparan</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()