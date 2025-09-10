import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for deployment
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io
import plotly.express as px
import pandas as pd
import os 
import traceback
import gc
import psutil
from captum.attr import Saliency, IntegratedGradients
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import EfficientNetClassifier

import warnings
warnings.filterwarnings('ignore')

# Set memory and computation limits for deployment
torch.set_num_threads(2)  # Limit CPU threads
if torch.cuda.is_available():
    torch.cuda.empty_cache()

st.set_page_config(
    page_title="AI Image Classifier with XAI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Add debug toggle and deployment info
DEBUG_MODE = st.sidebar.checkbox("Debug Mode", value=False)
DEPLOYMENT_MODE = st.sidebar.checkbox("Deployment Mode", value=True)

def debug_print(message):
    """Print debug messages only if debug mode is enabled"""
    if DEBUG_MODE:
        st.write(f"🐛 DEBUG: {message}")

def get_memory_usage():
    """Get current memory usage - simplified version without psutil"""
    try:
        # Alternative method without psutil
        import resource
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Convert to MB (different units on different systems)
        if os.name == 'posix':
            return memory_usage / 1024  # KB to MB on Unix
        else:
            return memory_usage / (1024 * 1024)  # Bytes to MB on Windows
    except Exception:
        return 0  # Return 0 if unable to get memory info

def cleanup_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    
    .memory-info {
        background: #f8f9fa; padding: 0.5rem; border-radius: 5px;
        font-size: 0.8rem; color: #666; margin: 0.5rem 0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model(model_path, device):
    """Load model with comprehensive error handling"""
    try:
        debug_print(f"Loading model from: {model_path}")
        debug_print(f"Using device: {device}")
        
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan: {model_path}")
            return None, None
        
        # Load with CPU if memory is limited
        if DEPLOYMENT_MODE:
            device = torch.device("cpu")
            debug_print("Forcing CPU usage for deployment mode")
        
        checkpoint = torch.load(model_path, map_location=device)
        debug_print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        class_names = checkpoint.get('class_names')
        num_classes = checkpoint.get('num_classes')
        
        if class_names is None:
            st.error("Checkpoint tidak berisi 'class_names'.")
            return None, None

        if num_classes is None:
            if class_names:
                num_classes = len(class_names)
                debug_print(f"num_classes derived from class_names: {num_classes}")
            else:
                st.error("Checkpoint tidak berisi 'num_classes' dan 'class_names'.")
                return None, None

        debug_print(f"Number of classes: {num_classes}")
        debug_print(f"Class names: {class_names}")

        model = EfficientNetClassifier(num_classes=num_classes).to(device)
        
        # Load state dict with error handling
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except KeyError:
            st.error("Checkpoint tidak berisi 'model_state_dict'.")
            return None, None
        except Exception as e:
            st.error(f"Error loading model state dict: {e}")
            return None, None
            
        model.eval()
        debug_print("Model loaded successfully")
        
        # Clear checkpoint from memory
        del checkpoint
        cleanup_memory()
        
        return model, class_names
        
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        if DEBUG_MODE:
            st.error(f"Full traceback: {traceback.format_exc()}")
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
        try:
            debug_print(f"Processing image of type: {type(image)}")
            
            if isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                image = Image.fromarray(image).convert('RGB')
            
            # Limit image size for deployment
            if DEPLOYMENT_MODE:
                max_size = (512, 512)
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
                debug_print(f"Image resized for deployment: {image.size}")
            
            debug_print(f"Image size: {image.size}")
            original_image = np.array(image)
            input_tensor = self.transform(image).unsqueeze(0)
            debug_print(f"Input tensor shape: {input_tensor.shape}")
            
            return input_tensor, original_image
            
        except Exception as e:
            st.error(f"Error in preprocess_image: {e}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def denormalize_tensor(self, tensor):
        return self.denormalize(tensor)

class ModelLoader:
    def __init__(self, model_path):
        # Force CPU for deployment mode
        if DEPLOYMENT_MODE:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug_print(f"Using device: {self.device}")
        # Call the cached function during initialization
        self.model, self.class_names = load_cached_model(model_path, self.device)

    def is_ready(self):
        """Check if model was successfully loaded."""
        return self.model is not None and self.class_names is not None

    def predict(self, input_tensor):
        if not self.is_ready():
            raise ValueError("Model belum dimuat atau gagal dimuat!")
        
        try:
            input_tensor = input_tensor.to(self.device)
            debug_print(f"Input tensor shape for prediction: {input_tensor.shape}")
            debug_print(f"Input tensor device: {input_tensor.device}")
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                debug_print(f"Model outputs shape: {outputs.shape}")
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
            
            debug_print(f"Predicted class: {predicted_class}")
            debug_print(f"Probabilities: {probabilities.cpu().numpy()[0]}")
            
            # Cleanup
            cleanup_memory()
            
            return predicted_class, probabilities.cpu().numpy()[0]
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            raise

class XAIVisualizer:
    def __init__(self, model, device):
        try:
            self.model = model
            self.device = device
            debug_print("Initializing XAI components...")
            
            # Initialize only essential XAI methods for deployment
            self.saliency = Saliency(model)
            debug_print("Saliency initialized")
            
            # Initialize Integrated Gradients with reduced steps for deployment
            self.integrated_gradients = IntegratedGradients(model)
            debug_print("Integrated Gradients initialized")
            
            # Initialize GradCAM methods with error handling
            try:
                target_layers = self._find_target_layers()
                debug_print(f"Target layers found: {len(target_layers)}")
                
                self.grad_cam = GradCAM(model=model, target_layers=target_layers)
                debug_print("GradCAM initialized successfully")
                
                # Skip ScoreCAM in deployment mode to save memory
                if not DEPLOYMENT_MODE:
                    try:
                        self.score_cam = ScoreCAM(model=model, target_layers=target_layers)
                        debug_print("ScoreCAM initialized successfully")
                    except Exception as e:
                        debug_print(f"ScoreCAM initialization failed: {e}")
                        self.score_cam = None
                else:
                    debug_print("Skipping ScoreCAM in deployment mode")
                    self.score_cam = None
                    
            except Exception as e:
                debug_print(f"CAM methods initialization failed: {e}")
                self.grad_cam = None
                self.score_cam = None
                
        except Exception as e:
            st.error(f"Error initializing XAI visualizer: {e}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _find_target_layers(self):
        """Find appropriate target layers for CAM methods"""
        target_layers = []
        
        try:
            # Try EfficientNet layers
            if hasattr(self.model, 'efficientnet'):
                if hasattr(self.model.efficientnet, 'features'):
                    target_layers.append(self.model.efficientnet.features[-1])
                    debug_print("Found EfficientNet features layer")
                elif hasattr(self.model.efficientnet, 'blocks'):
                    target_layers.append(self.model.efficientnet.blocks[-1])
                    debug_print("Found EfficientNet blocks layer")
            
            # If no layers found, try to find any conv layers
            if not target_layers:
                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                        target_layers = [module]
                        debug_print(f"Found fallback layer: {name}")
                        break
            
            if not target_layers:
                raise ValueError("No suitable target layers found for CAM methods")
                
            return target_layers
            
        except Exception as e:
            debug_print(f"Error finding target layers: {e}")
            raise
    
    def generate_saliency_map(self, input_tensor, target_class):
        try:
            debug_print(f"Generating saliency map for class {target_class}")
            
            # Clear memory before computation
            cleanup_memory()
            
            input_tensor = input_tensor.to(self.device)
            input_tensor.requires_grad_(True)
            
            attribution = self.saliency.attribute(input_tensor, target=target_class)
            result = attribution.squeeze().cpu().detach().numpy()
            debug_print(f"Saliency map shape: {result.shape}")
            
            # Clear intermediate tensors
            del attribution
            cleanup_memory()
            
            return result
            
        except Exception as e:
            st.error(f"Error in generate_saliency_map: {str(e)}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            cleanup_memory()
            return None
    
    def generate_integrated_gradients(self, input_tensor, target_class, steps=None):
        try:
            # Reduce steps for deployment
            if steps is None:
                steps = 10 if DEPLOYMENT_MODE else 50
                
            debug_print(f"Generating integrated gradients for class {target_class} with {steps} steps")
            
            # Clear memory before computation
            cleanup_memory()
            
            input_tensor = input_tensor.to(self.device)
            
            attribution = self.integrated_gradients.attribute(
                input_tensor, 
                target=target_class, 
                n_steps=steps
            )
            result = attribution.squeeze().cpu().detach().numpy()
            debug_print(f"Integrated gradients shape: {result.shape}")
            
            # Clear intermediate tensors
            del attribution
            cleanup_memory()
            
            return result
            
        except Exception as e:
            st.error(f"Error in generate_integrated_gradients: {str(e)}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            cleanup_memory()
            return None
    
    def generate_grad_cam(self, input_tensor, target_class):
        try:
            if self.grad_cam is None:
                st.warning("GradCAM not available")
                return None
                
            debug_print(f"Generating GradCAM for class {target_class}")
            
            # Clear memory before computation
            cleanup_memory()
            
            input_tensor = input_tensor.to(self.device)
            
            targets = [ClassifierOutputTarget(target_class)]
            grayscale_cam = self.grad_cam(input_tensor=input_tensor, targets=targets)
            result = grayscale_cam[0, :]
            debug_print(f"GradCAM shape: {result.shape}")
            
            # Clear intermediate data
            del grayscale_cam
            cleanup_memory()
            
            return result
            
        except Exception as e:
            st.error(f"Error in generate_grad_cam: {str(e)}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            cleanup_memory()
            return None
    
    def generate_score_cam(self, input_tensor, target_class):
        try:
            if self.score_cam is None:
                st.warning("ScoreCAM not available in deployment mode")
                return None
                
            debug_print(f"Generating ScoreCAM for class {target_class}")
            
            # Clear memory before computation
            cleanup_memory()
            
            if input_tensor.ndim == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            targets = [ClassifierOutputTarget(target_class)]
            
            grayscale_cam = self.score_cam(input_tensor=input_tensor, targets=targets)
            
            heatmap = grayscale_cam[0, :]
            heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
            debug_print(f"ScoreCAM shape: {heatmap.shape}")
            
            # Clear intermediate data
            del grayscale_cam
            cleanup_memory()
            
            return heatmap
            
        except Exception as e:
            st.error(f"Error in generate_score_cam: {str(e)}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            cleanup_memory()
            return None
    
    def create_heatmap_overlay(self, original_image, heatmap, alpha=0.4):
        try:
            if heatmap is None:
                return original_image
                
            debug_print(f"Creating heatmap overlay - Original: {original_image.shape}, Heatmap: {heatmap.shape}")
            
            # Limit image size for memory efficiency
            if DEPLOYMENT_MODE and original_image.shape[0] > 512:
                scale_factor = 512 / max(original_image.shape[:2])
                new_size = (int(original_image.shape[1] * scale_factor), 
                           int(original_image.shape[0] * scale_factor))
                original_image = cv2.resize(original_image, new_size)
                debug_print(f"Image resized to: {original_image.shape}")
            
            # Normalize heatmap
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Resize heatmap to match original image
            heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            
            # Create colored heatmap
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            
            # Normalize original image
            original_normalized = original_image.astype(np.float32) / 255.0
            
            # Blend images
            blended = (1 - alpha) * original_normalized + alpha * heatmap_colored
            
            return (blended * 255).astype(np.uint8)
            
        except Exception as e:
            st.error(f"Error creating heatmap overlay: {e}")
            if DEBUG_MODE:
                st.error(f"Full traceback: {traceback.format_exc()}")
            return original_image

def create_prediction_chart(probabilities, class_names):
    try:
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
        
    except Exception as e:
        st.error(f"Error creating prediction chart: {e}")
        return None

def display_xai_visualization(original_image, attribution, title, method_type='heatmap'):
    try:
        if attribution is None:
            st.warning(f"⚠️ Tidak dapat menghasilkan visualisasi untuk {title}")
            return
        
        debug_print(f"Displaying {title} - Attribution shape: {attribution.shape}")
        
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
            
            if method_type == 'heatmap':
                visualizer = st.session_state.xai_visualizer
                overlay = visualizer.create_heatmap_overlay(original_image, attribution)
                st.image(overlay, caption=f"{title} Visualization", use_container_width=True)
            else:
                # Handle gradient-based methods with memory optimization
                try:
                    if len(attribution.shape) == 3:  # (C, H, W)
                        attribution_normalized = np.transpose(attribution, (1, 2, 0))
                        attribution_gray = np.mean(np.abs(attribution_normalized), axis=2)
                    elif len(attribution.shape) == 2:  # (H, W)
                        attribution_gray = np.abs(attribution)
                    else:
                        st.error(f"Unexpected attribution shape: {attribution.shape}")
                        return
                    
                    # Create visualization with smaller figure size for deployment
                    fig_size = (6, 6) if DEPLOYMENT_MODE else (8, 8)
                    fig, ax = plt.subplots(figsize=fig_size, dpi=80)
                    ax.imshow(attribution_gray, cmap='hot', alpha=0.8)
                    ax.imshow(original_image, alpha=0.3)
                    ax.axis('off')
                    ax.set_title(f'{title} Visualization', fontsize=12, fontweight='bold')
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white', dpi=80)
                    buf.seek(0)
                    st.image(buf, use_container_width=True)
                    plt.close()
                    
                    # Clear buffer
                    buf.close()
                    
                except Exception as viz_error:
                    st.error(f"Error creating gradient visualization: {viz_error}")
                    
        # Force cleanup after each visualization
        cleanup_memory()
                
    except Exception as e:
        st.error(f"Error dalam visualisasi {title}: {str(e)}")
        if DEBUG_MODE:
            st.error(f"Full traceback: {traceback.format_exc()}")
        cleanup_memory()

def main():
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🔍 Chest X-ray Classification</div>
        <div class="header-subtitle">with Explainable AI (XAI) Visualization</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display memory info if in debug mode
    if DEBUG_MODE:
        memory_usage = get_memory_usage()
        st.markdown(f"""
        <div class="memory-info">
            Memory Usage: {memory_usage:.1f} MB | Deployment Mode: {DEPLOYMENT_MODE}
        </div>
        """, unsafe_allow_html=True)
    
    from pathlib import Path

    APP_DIR = Path(__file__).resolve().parent
    MODEL_PATH = APP_DIR / "efficientnet_b0_classifier.pth"
    
    debug_print(f"App directory: {APP_DIR}")
    debug_print(f"Model path: {MODEL_PATH}")

    # Initialize objects in session state (if they don't exist)
    if 'image_processor' not in st.session_state:
        debug_print("Initializing image processor")
        st.session_state.image_processor = ImageProcessor()
    
    # Initialize the model loader (this will automatically call the cached function)
    if 'model_loader' not in st.session_state:
        with st.spinner("Memuat model AI, harap tunggu..."):
            debug_print("Initializing model loader")
            st.session_state.model_loader = ModelLoader(MODEL_PATH)
            if st.session_state.model_loader.is_ready():
                st.success("✅ Model berhasil dimuat!")
            else:
                st.error("❌ Gagal memuat model!")

    # Check if the model is ready to use
    if st.session_state.model_loader.is_ready():
        # Initialize the XAI visualizer now that the model is confirmed to be loaded
        if 'xai_visualizer' not in st.session_state:
            try:
                with st.spinner("Menginisialisasi XAI visualizer..."):
                    debug_print("Initializing XAI visualizer")
                    st.session_state.xai_visualizer = XAIVisualizer(
                        st.session_state.model_loader.model,
                        st.session_state.model_loader.device
                    )
                    st.success("✅ XAI visualizer berhasil diinisialisasi!")
                    cleanup_memory()
            except Exception as e:
                st.error(f"Gagal menginisialisasi XAI: {e}")
                if DEBUG_MODE:
                    st.error(f"Full traceback: {traceback.format_exc()}")
                st.stop()

        # MAIN APPLICATION LOGIC
        st.markdown("""
        <div class="info-card">
            <h3>📸 Upload Gambar Chest X-ray</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih gambar...",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            try:
                # Process the uploaded image
                with st.spinner("Memproses gambar..."):
                    debug_print("Processing uploaded image")
                    original_image = Image.open(uploaded_file)
                    input_tensor, original_array = st.session_state.image_processor.preprocess_image(original_image)
                
                # Make prediction
                with st.spinner("Melakukan prediksi..."):
                    debug_print("Making prediction")
                    predicted_class, probabilities = st.session_state.model_loader.predict(input_tensor)
                    predicted_class_name = st.session_state.model_loader.class_names[predicted_class]
                    confidence = probabilities[predicted_class] * 100
                
                debug_print(f"Prediction: {predicted_class_name} ({confidence:.2f}%)")
                
                # Display results
                st.markdown("""
                <div class="results-container">
                    <h2>📊 Hasil Prediksi</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-container">
                    <div class="prediction-class">Prediksi: {predicted_class_name}</div>
                    <div class="prediction-confidence">Tingkat Kepercayaan: {confidence:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display original image
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(original_image, caption="Gambar yang diunggah", use_container_width=True)
                
                with col2:
                    # Display probability chart
                    fig = create_prediction_chart(probabilities, st.session_state.model_loader.class_names)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                # XAI Visualizations with deployment mode considerations
                st.markdown("""
                <div class="results-container">
                    <h2>🧠 Explainable AI (XAI) Analysis</h2>
                    <p>Visualisasi berikut menunjukkan bagian mana dari gambar yang paling berpengaruh dalam keputusan model:</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add option to enable/disable XAI methods
                xai_methods = st.multiselect(
                    "Pilih metode XAI yang ingin ditampilkan:",
                    ["Saliency Map", "Integrated Gradients", "Grad-CAM", "Score-CAM"],
                    default=["Saliency Map", "Grad-CAM"] if DEPLOYMENT_MODE else ["Saliency Map", "Integrated Gradients", "Grad-CAM"]
                )
                
                if st.button("🚀 Generate XAI Visualizations", type="primary"):
                    # Generate and display XAI visualizations with individual error handling
                    debug_print("Starting XAI visualizations")
                    
                    # Progress bar for better UX
                    progress_bar = st.progress(0)
                    
                    # Saliency Map
                    if "Saliency Map" in xai_methods:
                        try:
                            st.markdown("### 1. Saliency Map")
                            st.info("📍 **Saliency Map** menunjukkan pixel mana yang paling sensitif terhadap perubahan output model.")
                            
                            with st.spinner("Menghasilkan Saliency Map..."):
                                debug_print("Generating Saliency Map")
                                progress_bar.progress(25)
                                saliency_attr = st.session_state.xai_visualizer.generate_saliency_map(
                                    input_tensor, predicted_class
                                )
                                display_xai_visualization(
                                    original_array, saliency_attr, "Saliency Map", method_type='gradient'
                                )
                                cleanup_memory()
                                
                        except Exception as e:
                            st.error(f"❌ Error generating Saliency Map: {e}")
                            if DEBUG_MODE:
                                st.error(f"Full traceback: {traceback.format_exc()}")
                            cleanup_memory()
                    
                    # Integrated Gradients
                    if "Integrated Gradients" in xai_methods:
                        try:
                            st.markdown("### 2. Integrated Gradients")
                            st.info("📍 **Integrated Gradients** menghitung kontribusi setiap pixel dengan mengintegrasikan gradien sepanjang jalur dari baseline ke input.")
                            
                            with st.spinner("Menghasilkan Integrated Gradients..."):
                                debug_print("Generating Integrated Gradients")
                                progress_bar.progress(50)
                                ig_attr = st.session_state.xai_visualizer.generate_integrated_gradients(
                                    input_tensor, predicted_class
                                )
                                display_xai_visualization(
                                    original_array, ig_attr, "Integrated Gradients", method_type='gradient'
                                )
                                cleanup_memory()
                                
                        except Exception as e:
                            st.error(f"❌ Error generating Integrated Gradients: {e}")
                            if DEBUG_MODE:
                                st.error(f"Full traceback: {traceback.format_exc()}")
                            cleanup_memory()
                    
                    # GradCAM
                    if "Grad-CAM" in xai_methods:
                        try:
                            st.markdown("### 3. Grad-CAM")
                            st.info("📍 **Grad-CAM** menggunakan gradien dari layer konvolusi terakhir untuk menghasilkan heatmap yang menunjukkan area penting.")
                            
                            with st.spinner("Menghasilkan Grad-CAM..."):
                                debug_print("Generating Grad-CAM")
                                progress_bar.progress(75)
                                gradcam_attr = st.session_state.xai_visualizer.generate_grad_cam(
                                    input_tensor, predicted_class
                                )
                                display_xai_visualization(
                                    original_array, gradcam_attr, "Grad-CAM", method_type='heatmap'
                                )
                                cleanup_memory()
                                
                        except Exception as e:
                            st.error(f"❌ Error generating Grad-CAM: {e}")
                            if DEBUG_MODE:
                                st.error(f"Full traceback: {traceback.format_exc()}")
                            cleanup_memory()
                    
                    # ScoreCAM
                    if "Score-CAM" in xai_methods and not DEPLOYMENT_MODE:
                        try:
                            st.markdown("### 4. Score-CAM")
                            st.info("📍 **Score-CAM** menggunakan activation maps untuk menghasilkan visualisasi tanpa mengandalkan gradien.")
                            
                            with st.spinner("Menghasilkan Score-CAM..."):
                                debug_print("Generating Score-CAM")
                                progress_bar.progress(90)
                                scorecam_attr = st.session_state.xai_visualizer.generate_score_cam(
                                    input_tensor, predicted_class
                                )
                                display_xai_visualization(
                                    original_array, scorecam_attr, "Score-CAM", method_type='heatmap'
                                )
                                cleanup_memory()
                                
                        except Exception as e:
                            st.error(f"❌ Error generating Score-CAM: {e}")
                            if DEBUG_MODE:
                                st.error(f"Full traceback: {traceback.format_exc()}")
                            cleanup_memory()
                    elif "Score-CAM" in xai_methods and DEPLOYMENT_MODE:
                        st.warning("⚠️ Score-CAM dinonaktifkan dalam deployment mode untuk menghemat memori.")
                    
                    progress_bar.progress(100)
                    st.success("✅ Analisis XAI selesai!")
                    
                    # Final memory cleanup
                    cleanup_memory()
                    
                    if DEBUG_MODE:
                        final_memory = get_memory_usage()
                        st.info(f"Final memory usage: {final_memory:.1f} MB")
                
                # Add interpretation guide
                with st.expander("📖 Panduan Interpretasi XAI"):
                    st.markdown("""
                    ### Cara Membaca Visualisasi XAI:
                    
                    **🔥 Area Merah/Panas:**
                    - Menunjukkan bagian gambar yang SANGAT BERPENGARUH terhadap keputusan model
                    - Model fokus pada area ini untuk membuat prediksi
                    
                    **🟡 Area Kuning/Sedang:**
                    - Menunjukkan bagian gambar yang CUKUP BERPENGARUH
                    - Memberikan kontribusi sedang terhadap keputusan
                    
                    **🔵 Area Biru/Dingin:**
                    - Menunjukkan bagian gambar yang KURANG BERPENGARUH
                    - Model tidak terlalu memperhatikan area ini
                    
                    ### Perbedaan Metode:
                    - **Saliency Map:** Fokus pada perubahan gradien langsung
                    - **Integrated Gradients:** Lebih stabil dan akurat dalam atribusi
                    - **Grad-CAM:** Menunjukkan area luas yang penting untuk klasifikasi
                    - **Score-CAM:** Independen dari gradien, fokus pada aktivasi
                    """)
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses gambar: {str(e)}")
                if DEBUG_MODE:
                    st.error(f"Full traceback: {traceback.format_exc()}")
                st.error("Silakan coba lagi dengan gambar yang berbeda.")
                cleanup_memory()
    else:
        # If the model is not ready, display an error and stop the app
        st.error(f"❌ Gagal memuat model. Pastikan file '{MODEL_PATH}' ada dan tidak rusak.")
        st.info("Pastikan file model 'efficientnet_b0_classifier.pth' tersedia di direktori aplikasi.")
        
        # Troubleshooting section
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            ### Kemungkinan Masalah:
            1. **File model tidak ditemukan** - Pastikan file `efficientnet_b0_classifier.pth` ada di direktori yang sama
            2. **Model format salah** - Pastikan model disimpan dengan format yang benar
            3. **Memori tidak cukup** - Aktifkan 'Deployment Mode' di sidebar
            4. **Dependensi missing** - Pastikan semua library terinstall
            
            ### Solusi:
            - Aktifkan 'Debug Mode' di sidebar untuk informasi lebih detail
            - Aktifkan 'Deployment Mode' jika mengalami masalah memori
            - Cek log error untuk informasi spesifik
            """)
        
        st.stop()

if __name__ == "__main__":
    main()