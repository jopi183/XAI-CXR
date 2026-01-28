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
from captum.attr import Saliency, GuidedBackprop, DeepLift, IntegratedGradients
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus
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
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px;
        font-weight: 600;
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .title-text {
        color: #2c3e50;
        font-weight: 700;
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
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            class_names = checkpoint.get('class_names')
            num_classes = checkpoint.get('num_classes')
        else:
            state_dict = checkpoint
            classifier_key = None
            for key in state_dict.keys():
                if 'classifier' in key and 'weight' in key:
                    classifier_key = key
                    break
            
            if classifier_key:
                num_classes = state_dict[classifier_key].shape[0]
            else:
                st.error("Cannot determine number of classes from state_dict")
                return None, None
            
            class_names = [f"Class_{i}" for i in range(num_classes)]
        
        clean_state_dict = {
            k: v for k, v in state_dict.items() 
            if not k.endswith("total_ops") and not k.endswith("total_params") and not k.endswith("num_batches_tracked")
        }
        
        fixed_state_dict = {}
        for k, v in clean_state_dict.items():
            if k.startswith('efficientnet.'):
                new_key = k
            elif k.startswith('features.') or k.startswith('classifier.'):
                new_key = 'efficientnet.' + k
            else:
                new_key = k
            fixed_state_dict[new_key] = v
        
        model = EfficientNetClassifier(num_classes=num_classes).to(device)
        
        try:
            model.load_state_dict(fixed_state_dict, strict=True)
        except RuntimeError as e:
            st.warning("Attempting flexible model loading due to key mismatch...")
            alternative_state_dict = {}
            for k, v in clean_state_dict.items():
                if k.startswith('efficientnet.'):
                    new_key = k.replace('efficientnet.', '')
                else:
                    new_key = k
                alternative_state_dict[new_key] = v
            
            model.load_state_dict(alternative_state_dict, strict=False)
        
        model.eval()
        return model, class_names
        
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        import traceback
        st.error(traceback.format_exc())
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
        self.guided_backprop = GuidedBackprop(model)
        self.deeplift = DeepLift(model)
        self.integrated_gradients = IntegratedGradients(model)
        
        self.target_layers = [model.efficientnet.features[-1]]
    
    def generate_saliency_map(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        try:
            attribution = self.saliency.attribute(input_tensor, target=target_class)
            attr_np = attribution.squeeze().cpu().detach().numpy()
            attr_np = np.abs(attr_np).sum(axis=0)
            return attr_np
        except Exception as e:
            st.error(f"Error dalam generate_saliency_map: {str(e)}")
            return None
    
    def generate_guided_backprop(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_(True)
        try:
            attribution = self.guided_backprop.attribute(input_tensor, target=target_class)
            attr_np = attribution.squeeze().cpu().detach().numpy()
            attr_np = np.maximum(attr_np, 0)
            attr_np = np.transpose(attr_np, (1, 2, 0))
            attr_np = np.mean(attr_np, axis=2)
            return attr_np
        except Exception as e:
            st.error(f"Error dalam generate_guided_backprop: {str(e)}")
            return None
    
    def generate_deeplift(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        try:
            baseline = torch.zeros_like(input_tensor)
            attribution = self.deeplift.attribute(input_tensor, baseline, target=target_class)
            attr_np = attribution.squeeze().cpu().detach().numpy()
            attr_np = attr_np.sum(axis=0)
            return attr_np
        except Exception as e:
            st.error(f"Error dalam generate_deeplift: {str(e)}")
            return None
    
    def generate_integrated_gradients(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        try:
            baseline = torch.zeros_like(input_tensor)
            attribution = self.integrated_gradients.attribute(
                input_tensor, 
                baseline, 
                target=target_class,
                n_steps=50
            )
            attr_np = attribution.squeeze().cpu().detach().numpy()
            attr_np = np.mean(np.abs(attr_np), axis=0)
            return attr_np
        except Exception as e:
            st.error(f"Error dalam generate_integrated_gradients: {str(e)}")
            return None
    
    def generate_grad_cam(self, input_tensor, target_class):
        """Generate GradCAM - uses gradients to weight feature maps"""
        input_tensor = input_tensor.to(self.device)
        try:
            grad_cam = GradCAM(model=self.model, target_layers=self.target_layers)
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0, :]
        except Exception as e:
            st.error(f"Error dalam generate_grad_cam: {str(e)}")
            return None
    
    def generate_grad_cam_plus(self, input_tensor, target_class):
        input_tensor = input_tensor.to(self.device)
        try:
            grad_cam_plus = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = grad_cam_plus(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0, :]
        except Exception as e:
            st.error(f"Error dalam generate_grad_cam_plus: {str(e)}")
            return None
    
    def generate_score_cam(self, input_tensor, target_class):
        try:
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            score_cam = ScoreCAM(
                model=self.model, 
                target_layers=self.target_layers,
                use_cuda=torch.cuda.is_available()
            )
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = score_cam(input_tensor=input_tensor, targets=targets)
            return grayscale_cam[0, :]
        except Exception as e:
            st.error(f"Error dalam generate_score_cam: {str(e)}")
            return None
    
    def create_heatmap_overlay(self, original_image, heatmap, alpha=0.5, colormap='jet'):
        if heatmap is None:
            return original_image
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        cmap = plt.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]
        
        original_normalized = original_image.astype(np.float32) / 255.0
        
        blended = (1 - alpha) * original_normalized + alpha * heatmap_colored
        return (blended * 255).astype(np.uint8)
    
    def create_gradient_overlay(self, original_image, gradient_map, alpha=0.6):
        if gradient_map is None:
            return original_image
        
        gradient_map = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-8)
        
        gradient_resized = cv2.resize(gradient_map, (original_image.shape[1], original_image.shape[0]))
        
        cmap = plt.get_cmap('hot')
        gradient_colored = cmap(gradient_resized)[:, :, :3]
        
        original_normalized = original_image.astype(np.float32) / 255.0
        
        blended = (1 - alpha) * original_normalized + alpha * gradient_colored
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


def display_xai_visualization(original_image, attribution, title, method_type='heatmap', visualizer=None):
    """Display XAI visualization with proper distinction between methods"""
    if attribution is None:
        st.warning(f"Tidak dapat menghasilkan visualisasi untuk {title}")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""<div class="result-box">
            <h4 class="title-text">Original Image</h4>
        </div>""", unsafe_allow_html=True)
        st.image(original_image, caption="Original Image", use_container_width=True)
    
    with col2:
        st.markdown(f"""<div class="result-box">
            <h4 class="title-text">{title}</h4>
        </div>""", unsafe_allow_html=True)
        
        try:
            if method_type == 'heatmap':
                # CAM methods - use jet colormap
                overlay = visualizer.create_heatmap_overlay(
                    original_image, 
                    attribution, 
                    alpha=0.5, 
                    colormap='jet'
                )
                st.image(overlay, caption=f"{title} Overlay", use_container_width=True)
            else:
                # Gradient methods - use hot colormap
                overlay = visualizer.create_gradient_overlay(
                    original_image, 
                    attribution, 
                    alpha=0.6
                )
                st.image(overlay, caption=f"{title} Overlay", use_container_width=True)
            
            # Show statistics
            st.markdown(f"""
            **Attribution Statistics:**
            - Min: {attribution.min():.4f}
            - Max: {attribution.max():.4f}
            - Mean: {attribution.mean():.4f}
            - Std: {attribution.std():.4f}
            """)
            
        except Exception as e:
            st.error(f"Error dalam visualisasi {title}: {str(e)}")


def main():
    st.markdown("""<div class="result-box" style="text-align: center; padding: 30px;">
        <h1 class="title-text">Chest X-ray Classification</h1>
        <p style="color: #7f8c8d; font-size: 18px;">AI-powered medical image analysis with explainable AI</p>
    </div>""", unsafe_allow_html=True)
    
    from pathlib import Path
    APP_DIR = Path(__file__).resolve().parent
    MODEL_PATH = APP_DIR / "efficientnet_b0_classifier.pth"
    
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor()
    
    if 'model_loader' not in st.session_state:
        with st.spinner("Loading AI model..."):
            st.session_state.model_loader = ModelLoader(MODEL_PATH)
            
    if st.session_state.model_loader.is_ready():
        st.success("Model loaded successfully")
        
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
        
        st.markdown("""<div class="result-box">
            <h3 class="title-text">Upload Chest X-ray Image</h3>
        </div>""", unsafe_allow_html=True)
        
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
                
                st.markdown("""<div class="result-box">
                    <h3 class="title-text">Classification Results</h3>
                </div>""", unsafe_allow_html=True)
                
                st.markdown(f"""<div class="result-box" style="background-color: #e8f5e9;">
                    <h2 style="color: #2c3e50; margin: 0;">Prediction: {predicted_class_name}</h2>
                    <h3 style="color: #27ae60; margin: 10px 0 0 0;">Confidence: {confidence:.2f}%</h3>
                </div>""", unsafe_allow_html=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(original_image, caption="Uploaded Image", use_container_width=True)
                
                with col2:
                    fig = create_prediction_chart(probabilities, st.session_state.model_loader.class_names)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""<div class="result-box">
                    <h3 class="title-text">Explainable AI Analysis</h3>
                </div>""", unsafe_allow_html=True)
                
                
                st.markdown("""<div class="result-box">
                    <h4 class="title-text">Select Visualization Method</h4>
                </div>""", unsafe_allow_html=True)
                
                col_method1, col_method2 = st.columns(2)
                
                with col_method1:
                    st.markdown("**Gradient-based Methods:**")
                    gradient_methods = [
                        "Saliency Map",
                        "Guided Backpropagation",
                        "DeepLIFT",
                        "Integrated Gradients"
                    ]
                
                with col_method2:
                    st.markdown("**CAM-based Methods:**")
                    cam_methods = [
                        "Grad-CAM",
                        "Grad-CAM++",
                        "Score-CAM"
                    ]
                
                all_methods = gradient_methods + cam_methods
                selected_method = st.radio(
                    "Choose a method:",
                    all_methods,
                    index=0
                )
                
                with st.spinner(f"Generating {selected_method} visualization..."):
                    if selected_method == "Saliency Map":
                        attribution = st.session_state.xai_visualizer.generate_saliency_map(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Saliency Map", 
                            method_type='gradient',
                            visualizer=st.session_state.xai_visualizer
                        )
                    
                    elif selected_method == "Guided Backpropagation":
                        attribution = st.session_state.xai_visualizer.generate_guided_backprop(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Guided Backpropagation", 
                            method_type='gradient',
                            visualizer=st.session_state.xai_visualizer
                        )
                    
                    elif selected_method == "DeepLIFT":
                        attribution = st.session_state.xai_visualizer.generate_deeplift(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "DeepLIFT", 
                            method_type='gradient',
                            visualizer=st.session_state.xai_visualizer
                        )
                    
                    elif selected_method == "Integrated Gradients":
                        attribution = st.session_state.xai_visualizer.generate_integrated_gradients(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Integrated Gradients", 
                            method_type='gradient',
                            visualizer=st.session_state.xai_visualizer
                        )
                    
                    elif selected_method == "Grad-CAM":
                        attribution = st.session_state.xai_visualizer.generate_grad_cam(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Grad-CAM", 
                            method_type='heatmap',
                            visualizer=st.session_state.xai_visualizer
                        )
                    
                    elif selected_method == "Grad-CAM++":
                        attribution = st.session_state.xai_visualizer.generate_grad_cam_plus(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Grad-CAM++", 
                            method_type='heatmap',
                            visualizer=st.session_state.xai_visualizer
                        )
                    
                    elif selected_method == "Score-CAM":
                        attribution = st.session_state.xai_visualizer.generate_score_cam(
                            input_tensor, predicted_class
                        )
                        display_xai_visualization(
                            original_array, attribution, "Score-CAM", 
                            method_type='heatmap',
                            visualizer=st.session_state.xai_visualizer
                        )
                
                st.success(f"{selected_method} visualization generated successfully")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    else:
        st.error(f"Failed to load model. Please ensure '{MODEL_PATH}' exists and is not corrupted.")
        st.info("Make sure the model file 'efficientnet_b0_classifier.pth' is available in the application directory.")
        st.stop()


if __name__ == "__main__":
    main()