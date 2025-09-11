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
    
    .xai-selection-container {
        background: linear-gradient(145deg, #f8f9ff, #ffffff);
        padding: 1.5rem; border-radius: 12px;
        border: 2px solid #667eea; margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
    }
    
    .xai-description {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        margin-top: 1rem; border-left: 4px solid #667eea;
        font-size: 0.95rem; color: #495057;
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
    
    .control-buttons {
        margin: 1.5rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        text-align: center;
    }
    
    .button-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .stRadio > div {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .stRadio > div > label {
        background: white;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        border: 2px solid #e1e8ed;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: #f8f9ff;
        transform: translateY(-1px);
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-color: #667eea;
    }
    
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        min-width: 120px !important;
        padding: 0.5rem 1.5rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #28a745, #20c997) !important;
        color: white !important;
    }
    
    .reset-button {
        background: linear-gradient(135deg, #dc3545, #c82333) !important;
        color: white !important;
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
            <div class="xai-title">Gambar Asli</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    with col2:
        st.markdown(f"""
        <div class="xai-card">
            <div class="xai-title">{title} Visualization</div>
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
                ax.set_title(f'{title} Visualization', fontsize=14, fontweight='bold')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close()
        except Exception as e:
            st.error(f"Error dalam visualisasi {title}: {str(e)}")

def get_xai_description(method):
    descriptions = {
        "Saliency Map": "Menunjukkan piksel mana yang paling berkontribusi terhadap prediksi dengan menghitung gradien dari output terhadap input.",
        "Integrated Gradients": "Metode yang lebih robust dengan mengintegrasikan gradien sepanjang jalur dari baseline ke input untuk mengurangi noise.",
        "Grad-CAM": "Menggunakan gradien dari layer konvolusi terakhir untuk menghasilkan heatmap lokalisasi yang kasar namun spesifik kelas.",
        "Score-CAM": "Menggunakan forward pass untuk menghasilkan heatmap tanpa bergantung pada gradien, memberikan visualisasi yang lebih stabil."
    }
    return descriptions.get(method, "")

def reset_session_state():
    """Reset semua session state untuk memulai dari awal"""
    keys_to_keep = ['image_processor', 'model_loader', 'xai_visualizer']
    keys_to_remove = [key for key in st.session_state.keys() if key not in keys_to_keep]
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Reset specific states
    st.session_state.prediction_done = False
    st.session_state.xai_submitted = False

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

    # Initialize session state flags
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'xai_submitted' not in st.session_state:
        st.session_state.xai_submitted = False

    # Initialize objects in session state
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor()
    
    if 'model_loader' not in st.session_state:
        with st.spinner("Memuat model AI, harap tunggu..."):
            st.session_state.model_loader = ModelLoader(MODEL_PATH)
            if st.session_state.model_loader.is_ready():
                st.success("✅ Model berhasil dimuat!")
            else:
                st.error(f"❌ Gagal memuat model. Pastikan file '{MODEL_PATH}' ada dan tidak rusak.")
                st.info("Pastikan file model 'efficientnet_b0_classifier.pth' tersedia di direktori aplikasi.")
                st.stop()

    # Initialize XAI visualizer only when needed
    if st.session_state.model_loader.is_ready() and 'xai_visualizer' not in st.session_state:
        try:
            with st.spinner("Menginisialisasi XAI visualizer..."):
                st.session_state.xai_visualizer = XAIVisualizer(
                    st.session_state.model_loader.model,
                    st.session_state.model_loader.device
                )
                st.success("✅ XAI visualizer berhasil diinisialisasi!")
        except Exception as e:
            st.error(f"Gagal menginisialisasi XAI: {e}")
            st.stop()

    # MAIN APPLICATION LOGIC
    st.markdown("""
    <div class="info-card">
        <h3>📸 Upload Gambar Chest X-ray</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset button - always available
    if st.button("🔄 Reset Aplikasi", key="reset_app"):
        reset_session_state()
        st.rerun()
    
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file is not None:
        try:
            # Process image and make prediction
            if not st.session_state.prediction_done:
                with st.spinner("Memproses gambar dan melakukan prediksi..."):
                    original_image = Image.open(uploaded_file)
                    input_tensor, original_array = st.session_state.image_processor.preprocess_image(original_image)
                    
                    # Store processed data in session state
                    st.session_state.original_image = original_image
                    st.session_state.original_array = original_array
                    st.session_state.input_tensor = input_tensor
                    
                    # Make prediction
                    predicted_class, probabilities = st.session_state.model_loader.predict(input_tensor)
                    predicted_class_name = st.session_state.model_loader.class_names[predicted_class]
                    confidence = probabilities[predicted_class] * 100
                    
                    # Store prediction results
                    st.session_state.predicted_class = predicted_class
                    st.session_state.predicted_class_name = predicted_class_name
                    st.session_state.confidence = confidence
                    st.session_state.probabilities = probabilities
                    st.session_state.prediction_done = True

            # Display prediction results
            if st.session_state.prediction_done:
                st.markdown("""
                <div class="results-container">
                    <h2>📊 Hasil Prediksi</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-container">
                    <div class="prediction-class">Prediksi: {st.session_state.predicted_class_name}</div>
                    <div class="prediction-confidence">Tingkat Kepercayaan: {st.session_state.confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display original image and chart
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(st.session_state.original_image, caption="Gambar yang diunggah", use_container_width=True)
                
                with col2:
                    fig = create_prediction_chart(st.session_state.probabilities, st.session_state.model_loader.class_names)
                    st.plotly_chart(fig, use_container_width=True)
                
                # XAI Section
                st.markdown("""
                <div class="results-container">
                    <h2>🧠 Explainable AI (XAI) Analysis</h2>
                    <p>Pilih metode visualisasi untuk memahami bagian mana dari gambar yang paling berpengaruh dalam keputusan model:</p>
                </div>
                """, unsafe_allow_html=True)
                
                # XAI Method Selection
                st.markdown("""
                <div class="xai-selection-container">
                    <h4>🔬 Pilih Metode XAI Visualization</h4>
                </div>
                """, unsafe_allow_html=True)
                
                xai_options = [
                    "Saliency Map",
                    "Integrated Gradients", 
                    "Grad-CAM",
                    "Score-CAM"
                ]
                
                selected_method = st.radio(
                    "Metode Visualisasi:",
                    xai_options,
                    index=0,
                    help="Pilih salah satu metode untuk melihat visualisasi XAI"
                )
                
                # Display description
                description = get_xai_description(selected_method)
                if description:
                    st.markdown(f"""
                    <div class="xai-description">
                        <strong>Tentang {selected_method}:</strong><br>
                        {description}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Control buttons
                st.markdown("""
                <div class="control-buttons">
                    <h4>🎯 Kontrol XAI Processing</h4>
                    <p>Tekan tombol "Analyze XAI" untuk memproses visualisasi explainable AI</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col2:
                    if st.button("🚀 Analyze XAI", type="primary", use_container_width=True):
                        st.session_state.xai_submitted = True
                        st.session_state.selected_method = selected_method
                
                # Display XAI visualization if submitted
                if st.session_state.xai_submitted and st.session_state.get('selected_method') == selected_method:
                    with st.spinner(f"Menghasilkan visualisasi {selected_method}..."):
                        if selected_method == "Saliency Map":
                            attribution = st.session_state.xai_visualizer.generate_saliency_map(
                                st.session_state.input_tensor, st.session_state.predicted_class
                            )
                            display_xai_visualization(
                                st.session_state.original_array, attribution, "Saliency Map", method_type='gradient'
                            )
                        
                        elif selected_method == "Integrated Gradients":
                            attribution = st.session_state.xai_visualizer.generate_integrated_gradients(
                                st.session_state.input_tensor, st.session_state.predicted_class
                            )
                            display_xai_visualization(
                                st.session_state.original_array, attribution, "Integrated Gradients", method_type='gradient'
                            )
                        
                        elif selected_method == "Grad-CAM":
                            attribution = st.session_state.xai_visualizer.generate_grad_cam(
                                st.session_state.input_tensor, st.session_state.predicted_class
                            )
                            display_xai_visualization(
                                st.session_state.original_array, attribution, "Grad-CAM", method_type='heatmap'
                            )
                        
                        elif selected_method == "Score-CAM":
                            attribution = st.session_state.xai_visualizer.generate_score_cam(
                                st.session_state.input_tensor, st.session_state.predicted_class
                            )
                            display_xai_visualization(
                                st.session_state.original_array, attribution, "Score-CAM", method_type='heatmap'
                            )
                    
                    st.success(f"✅ Visualisasi {selected_method} berhasil dibuat!")
                    
                    # Reset XAI submission state when method changes
                    if st.session_state.get('selected_method') != selected_method:
                        st.session_state.xai_submitted = False
                
                # Additional control buttons
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Pilih Gambar Lain", use_container_width=True):
                        reset_session_state()
                        st.rerun()
                
                with col2:
                    if st.button("🆕 Analisis XAI Baru", use_container_width=True):
                        st.session_state.xai_submitted = False
                        st.rerun()
                        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {str(e)}")
            st.error("Silakan coba lagi dengan gambar yang berbeda.")

if __name__ == "__main__":
    main()