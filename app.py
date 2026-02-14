"""
HARE Skin Lesion Classifier - Streamlit Web Application

Final Year Research Project: "HARE: Hybrid Adversarially-Robust Ensemble for Skin Cancer Detection"

Features:
1. Binary classification (Melanoma vs Nevus) with clinical risk banners
2. Grad-CAM explainability with attention dispersion score
3. FGSM adversarial robustness demonstration
4. Hardcoded A100 training statistics and research context

Author: Final Year Project Student
Institution: Informatics Institute of Technology, Sri Lanka
Date: February 2026
"""

import os
from typing import Tuple

import streamlit as st
import torch
from PIL import Image

from hare_model import HARE_Ensemble
from utils import (
    prepare_image,
    denormalize_image,
    predict,
    GradCAM,
    overlay_heatmap,
    fgsm_attack,
    get_risk_banner,
    smart_load_weights,
    get_device,
    CLASS_LABELS,
    CLASS_DIAGNOSIS,
)


# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="HARE Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================
# Sidebar: Research Statistics & About
# ============================
with st.sidebar:
    st.markdown("---")
    st.header("🔬 HARE Research Results")
    st.markdown(
        """
        **A100 Training Results** (ISIC 2019 Dataset)
        
        Trained on binary classification: Melanoma (Malignant) vs Nevus (Benign)
        """
    )
    
    # Statistics in metric cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Clean Accuracy", "84.9%", "Test Set")
    with col2:
        st.metric("PGD Robustness", "95.0%", "ε=8/255")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Baseline Robustness", "28.4%", "ResNet50 Only")
    with col4:
        st.metric("CAR Score", "0.7295", "Clean-Adv Ratio")
    
    st.markdown("---")
    st.markdown(
        """
        **Why HARE Outperforms Baseline:**
        
        🔴 **Baseline (ResNet50 only):**
        - Only texture features
        - Vulnerable to adversarial attacks
        - 28.4% PGD robustness (ε=8/255)
        
        🟢 **HARE (ResNet50 + ViT):**
        - Hybrid feature fusion (texture + structure)
        - Evolved fusion layer via **OATGA** (Genetic Algorithm)
        - **95.0% PGD robustness** (ε=8/255)
        - **3.3× more robust** than baseline
        """
    )
    
    st.markdown("---")
    st.header("📚 About HARE & OATGA")
    
    with st.expander("HARE Architecture", expanded=False):
        st.markdown(
            """
            **HARE = Hybrid Attention ResNet Ensemble**
            
            Combines two complementary architectures:
            
            1. **ResNet50 (CNN Branch)**
               - Texture & edge detection
               - Local feature extraction
               - 2048-dimensional features
            
            2. **Vision Transformer (ViT) Branch**
               - Global structure understanding
               - Long-range spatial relationships
               - 768-dimensional features
            
            3. **GA-Optimized Fusion Layer**
               - Concatenates 2816 features → 512-dim latent space
               - Parameters evolved via genetic algorithm
               - Balances accuracy (30%) vs robustness (70%)
            
            4. **Binary Classifier**
               - 512-dim latent → 2 classes (Nevus, Melanoma)
               - Dropout (0.5) for regularization
            """
        )
    
    with st.expander("OATGA Optimization", expanded=False):
        st.markdown(
            """
            **OATGA = Optimization-Augmented Training with Genetic Algorithm**
            
            Two-phase training process:
            
            **Phase A: Adversarial Training**
            - Train HARE on both clean and adversarially perturbed images
            - Minimize cross-entropy loss on adversarial examples
            - Produces hare_best.pth checkpoint
            
            **Phase B: Genetic Algorithm Evolution**
            - Search space: Fusion layer parameters (512 × 2048 + 512 × 768)
            - Objective: Maximize fitness = 0.3 × clean_acc + 0.7 × robust_acc
            - Population-based search (crossover, mutation, selection)
            - Produces hare_final.pth (best evolved model)
            
            **Result:** Model optimized for both accuracy and robustness
            """
        )
    
    st.markdown("---")
    st.markdown(
        """
        **Note:** This demo uses binary weights (Melanoma vs Nevus).
        Results extrapolated from multi-class (8-class) ISIC 2019 training.
        """
    )


# ============================
# Model Loading (Cached)
# ============================
@st.cache_resource(show_spinner=False)
def load_model() -> Tuple[HARE_Ensemble, torch.device, str]:
    """Load binary HARE_Ensemble model with weights (cached)."""
    device = get_device()
    model = HARE_Ensemble(num_classes=2).to(device)
    model.eval()
    
    base_dir = os.path.dirname(__file__)
    ok, message = smart_load_weights(model, base_dir)
    return model, device, message


# ============================
# Main UI
# ============================
st.title("🔬 HARE Skin Lesion Classifier")
st.markdown(
    """
    **Hybrid Adversarially-Robust Ensemble for Skin Cancer Detection**
    
    Binary classification system for melanoma detection with explainability and robustness verification.
    
    📊 **Features:**
    - High accuracy classification (84.9%)
    - Visual explainability via Grad-CAM with attention concentration score
    - Adversarial robustness demonstration (FGSM attack simulation)
    - Clinical risk stratification (Malignant vs Benign)
    """
)

# Load model
model, device, weight_message = load_model()

# Display weight loading status
if "Loaded" in weight_message:
    st.success(weight_message)
else:
    st.warning(weight_message)

st.markdown("---")

# ============================
# Image Upload
# ============================
uploaded_file = st.file_uploader(
    "📸 Upload a skin lesion image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Recommended: 224×224 resolution or larger. Clear lighting, full lesion visible.",
)

if uploaded_file is None:
    st.info("👆 Please upload a skin lesion image to begin classification.")
    st.stop()

# Load and display image
image = Image.open(uploaded_file).convert("RGB")

# ============================
# Inference & Prediction
# ============================
image_tensor = prepare_image(image, device)
pred_idx, confidence, probs = predict(model, image_tensor)
pred_label = CLASS_LABELS[pred_idx]
diagnosis = CLASS_DIAGNOSIS.get(pred_label, pred_label)

# ============================
# Layout: Original Image | Prediction & Risk Banner
# ============================
col_img, col_pred = st.columns(2, gap="large")

with col_img:
    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)

with col_pred:
    st.subheader("🔍 Prediction Result")
    
    # Diagnosis metric
    st.metric(label="Classification", value=diagnosis, delta=None)
    
    # Confidence progress bar
    st.write("**Confidence Score:**")
    st.progress(confidence)
    st.caption(f"{confidence * 100:.1f}%")
    
    # Risk banner
    banner_text, banner_style = get_risk_banner(pred_label)
    if banner_style == "error":
        st.error(banner_text)
    else:
        st.success(banner_text)
    
    # Probability distribution
    st.write("**Class Probabilities:**")
    for label, prob in zip(CLASS_LABELS, probs):
        diagnosis_name = CLASS_DIAGNOSIS[label]
        st.write(f"  {diagnosis_name}: **{prob * 100:.1f}%**")

st.markdown("---")

# ============================
# Novelty Features: Tabs for Explainability & Robustness
# ============================
tab1, tab2 = st.tabs(["🔥 Explainability (Grad-CAM)", "🛡️ Robustness (FGSM Attack)"])

# -------- TAB 1: Grad-CAM --------
with tab1:
    st.markdown(
        """
        **Grad-CAM (Gradient-weighted Class Activation Map)**
        
        Visualizes which image regions the model focused on for its decision.
        Red regions = high importance | Blue regions = low importance
        """
    )
    
    try:
        grad_cam = GradCAM(model)
        heatmap, dispersion_score = grad_cam.generate(image_tensor, pred_idx)
        overlay = overlay_heatmap(image, heatmap, alpha=0.45)
        
        # Side-by-side visualization
        cam_col1, cam_col2 = st.columns(2, gap="medium")
        
        with cam_col1:
            st.subheader("Heatmap Overlay")
            st.image(overlay, use_container_width=True)
            st.caption("Red = high attention | Blue = low attention")
        
        with cam_col2:
            st.subheader("Raw Heatmap")
            st.image(heatmap, use_container_width=True, clamp=True)
            st.caption("Grayscale: 0 (dark) to 1 (bright)")
        
        # Dispersion interpretation
        st.write("**Attention Concentration Score:**")
        st.metric(
            label="Dispersion Score",
            value=f"{dispersion_score:.3f}",
            delta="Higher = More Focused",
        )
        
        if dispersion_score > 0.5:
            st.success(
                f"✅ **Focused Attention** ({dispersion_score:.1%}): Model concentrates on specific lesion regions. "
                "This demonstrates interpretable decision-making."
            )
        else:
            st.warning(
                f"⚠️ **Dispersed Attention** ({dispersion_score:.1%}): Model attends to many regions. "
                "Verify if background features are influencing the decision."
            )
    
    except Exception as exc:
        st.error(f"❌ Grad-CAM generation failed: {exc}")

# -------- TAB 2: FGSM Attack Robustness --------
with tab2:
    st.markdown(
        """
        **FGSM Adversarial Attack Simulation**
        
        Tests model robustness by adding imperceptible noise optimized to fool the classifier.
        The adversarial image looks normal to humans but challenges the model.
        
        **Interpretation:**
        - ✅ Prediction **unchanged** → Model is **robust** to this attack
        - ⚠️ Prediction **changed** → Model is **vulnerable** (shows area for improvement)
        """
    )
    
    if st.button("🎯 Simulate FGSM Attack", use_container_width=True):
        st.markdown("Computing adversarial example (ε=0.03 in normalized space)...")
        
        # Generate adversarial image
        label_tensor = torch.tensor([pred_idx], device=device)
        adv_tensor = fgsm_attack(model, image_tensor, label_tensor, epsilon=0.03)
        
        # Predict on adversarial image
        adv_pred_idx, adv_confidence, adv_probs = predict(model, adv_tensor)
        adv_label = CLASS_LABELS[adv_pred_idx]
        adv_diagnosis = CLASS_DIAGNOSIS.get(adv_label, adv_label)
        
        # Denormalize adversarial image for display
        adv_image_np = denormalize_image(adv_tensor)
        adv_image_pil = Image.fromarray((adv_image_np * 255).astype("uint8"))
        
        st.success("✅ Adversarial example generated!")
        
        # -------- Before vs After Comparison --------
        st.subheader("Attack Comparison")
        
        comparison_col1, comparison_col2 = st.columns(2, gap="large")
        
        with comparison_col1:
            st.write("**Clean Image (Original)**")
            st.image(image, use_container_width=True)
            st.metric(label="Diagnosis", value=diagnosis)
            st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
            risk_text, risk_style = get_risk_banner(pred_label)
            if risk_style == "error":
                st.error(risk_text)
            else:
                st.success(risk_text)
        
        with comparison_col2:
            st.write("**Adversarial Image (Attacked)**")
            st.image(adv_image_pil, use_container_width=True)
            st.metric(label="Diagnosis", value=adv_diagnosis)
            st.metric(label="Confidence", value=f"{adv_confidence * 100:.1f}%")
            adv_risk_text, adv_risk_style = get_risk_banner(adv_label)
            if adv_risk_style == "error":
                st.error(adv_risk_text)
            else:
                st.success(adv_risk_text)
        
        # -------- Robustness Verdict --------
        st.markdown("---")
        st.subheader("🎯 Robustness Verdict")
        
        if pred_label == adv_label:
            st.success(
                f"✅ **ROBUST**: Prediction remained **{pred_label}** (Confidence: {confidence*100:.1f}% → {adv_confidence*100:.1f}%)\n\n"
                f"The model maintained its decision despite adversarial perturbation. "
                f"This demonstrates robustness from **OATGA** optimization (95.0% PGD robustness)."
            )
        else:
            st.error(
                f"⚠️ **VULNERABLE**: Prediction changed **{pred_label}** → **{adv_label}**\n\n"
                f"The adversarial perturbation successfully fooled the model. "
                f"This would indicate insufficient adversarial training (unlike HARE's 95.0% PGD robustness on full dataset)."
            )
        
        # Detailed probability comparison
        st.write("**Probability Distribution Comparison:**")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.write("**Clean Predictions:**")
            for label, prob in zip(CLASS_LABELS, probs):
                st.write(f"  {label}: {prob * 100:.1f}%")
        
        with prob_col2:
            
            st.write("**Adversarial Predictions:**")
            for label, prob in zip(CLASS_LABELS, adv_probs):
                st.write(f"  {label}: {prob * 100:.1f}%")

st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This application is for research and demonstration purposes only.
    It is **not** a medical device and should **not** be used as a primary diagnostic tool.
    Always consult a qualified dermatologist for clinical diagnosis and treatment.
    """
)
