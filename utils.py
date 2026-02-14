"""
Utility functions for HARE Skin Lesion Classifier:
- Image preprocessing and tensor conversion
- Prediction and confidence calculation
- Grad-CAM visualization with dispersion score (attention concentration metric)
- FGSM adversarial attack generation
- Risk stratification logic
"""

import os
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.cm as cm

from hare_model import HARE_Ensemble


# ============================
# Configuration Constants
# ============================
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Binary classification labels
CLASS_LABELS = ["NV", "MEL"]  # 0=Nevus (Benign), 1=Melanoma (Malignant)
CLASS_DIAGNOSIS: Dict[str, str] = {
    "NV": "Melanocytic Nevus (Benign)",
    "MEL": "Melanoma (Malignant)",
}


# ============================
# Image Preprocessing
# ============================
preprocess = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def prepare_image(image: Image.Image, device: torch.device) -> torch.Tensor:
    """
    Preprocess a PIL image and convert to normalized tensor.
    
    Args:
        image (Image.Image): Input PIL image (RGB)
        device (torch.device): Target device (CPU or CUDA)
    
    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, 224, 224)
    """
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return image_tensor


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert normalized tensor back to displayable numpy array [0, 1].
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor of shape (1, 3, 224, 224)
    
    Returns:
        np.ndarray: Denormalized image array of shape (224, 224, 3)
    """
    image_tensor = image_tensor.squeeze(0).detach().cpu()
    image_tensor = image_tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    image_tensor = torch.clamp(image_tensor, 0, 1).permute(1, 2, 0).numpy()
    return image_tensor


# ============================
# Prediction & Inference
# ============================
def predict(model: nn.Module, image_tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
    """
    Run model inference and return class prediction with confidence.
    
    Args:
        model (nn.Module): HARE_Ensemble model in eval mode
        image_tensor (torch.Tensor): Preprocessed image tensor
    
    Returns:
        Tuple[int, float, np.ndarray]:
            - pred_idx (int): Predicted class index (0=Nevus, 1=Melanoma)
            - confidence (float): Softmax probability of predicted class [0, 1]
            - probs (np.ndarray): Full softmax probabilities for all classes
    """
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    return pred_idx, confidence, probs


# ============================
# Grad-CAM with Dispersion Score
# ============================
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for visual explainability.
    
    Hooks into ResNet50's final convolutional layer (layer4[-1]) to extract:
    - Activations: Final feature maps (before global avg pooling)
    - Gradients: Backpropagated gradients w.r.t. class score
    
    Computes class-specific heatmaps showing which image regions influenced the prediction.
    
    Citation: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via 
              Gradient-based Localization", ICCV 2017
    """
    
    def __init__(self, model: HARE_Ensemble):
        """
        Initialize Grad-CAM with hooks on ResNet50 layer4[-1].
        
        Args:
            model (HARE_Ensemble): Target model (must be in eval mode)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        target_layer = self.model.cnn.layer4[-1]  # Final residual block
        
        def forward_hook(module, input, output):
            """Capture activations during forward pass."""
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            """Capture gradients during backward pass."""
            self.gradients = grad_output[0]
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> Tuple[np.ndarray, float]:
        """
        Generate Grad-CAM heatmap for a specific class.
        
        Args:
            input_tensor (torch.Tensor): Preprocessed image tensor
            class_idx (int): Target class index for visualization
        
        Returns:
            Tuple[np.ndarray, float]:
                - heatmap (np.ndarray): Normalized heatmap [0, 1] of shape (14, 14)
                - dispersion_score (float): Concentration metric [0, 1]
                  0.0 = dispersed attention (spread across image)
                  1.0 = focused attention (concentrated on few regions)
        """
        # Zero gradients and forward pass
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, class_idx]
        score.backward(retain_graph=True)
        
        # Extract hooks
        gradients = self.gradients
        activations = self.activations
        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")
        
        # Compute weights: global average pooling of gradients across spatial dimensions
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Compute CAM: weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # ReLU to keep only positive contributions
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        
        # Compute dispersion score (attention concentration)
        dispersion_score = self._compute_dispersion_score(cam)
        
        return cam, dispersion_score
    
    @staticmethod
    def _compute_dispersion_score(heatmap: np.ndarray) -> float:
        """
        Compute dispersion score: measure of attention concentration.
        
        Strategy: Calculate percentage of pixels above median heatmap value.
        - High score (>0.5) = focused attention (concentrated on few regions)
        - Low score (<0.5) = dispersed attention (spread across many regions)
        
        Interpretation:
          - Focused: Model makes decision based on specific lesion features (good)
          - Dispersed: Model attends to many regions, possibly including background (questionable)
        
        Args:
            heatmap (np.ndarray): Normalized heatmap [0, 1]
        
        Returns:
            float: Dispersion score [0, 1]
        """
        threshold = np.median(heatmap)
        high_attention_pixels = (heatmap > threshold).sum()
        total_pixels = heatmap.size
        dispersion_score = high_attention_pixels / total_pixels
        return float(dispersion_score)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Overlay Grad-CAM heatmap on original image using Jet colormap.
    
    Red regions = high attention, Blue regions = low attention
    
    Args:
        image (Image.Image): Original PIL image (RGB)
        heatmap (np.ndarray): Normalized heatmap [0, 1]
        alpha (float): Blending weight for heatmap [0, 1]. Default: 0.45
    
    Returns:
        Image.Image: Overlay image (same size as input)
    """
    # Resize heatmap to match image size
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size))
    
    # Apply Jet colormap: heatmap values → RGB colors
    heatmap_color = cm.get_cmap("jet")(heatmap_resized / 255.0)[:, :, :3]
    heatmap_color = (heatmap_color * 255).astype(np.uint8)
    
    # Blend: alpha * heatmap + (1 - alpha) * original
    image_np = np.array(image).astype(np.uint8)
    overlay = (alpha * heatmap_color + (1 - alpha) * image_np).astype(np.uint8)
    
    return Image.fromarray(overlay)


# ============================
# FGSM Adversarial Attack
# ============================
def fgsm_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    epsilon: float = 0.03,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    
    Generates a small perturbation that maximizes classification loss:
    - x_adv = x + ε × sign(∇_x L(f(x), y))
    - Where ε is scaled by ImageNet std for normalized image space
    
    The perturbation is imperceptible to humans but can fool the model,
    demonstrating either vulnerability or robustness depending on whether
    the prediction changes.
    
    Args:
        model (nn.Module): Model to attack (in eval mode)
        image_tensor (torch.Tensor): Normalized input image (B, 3, 224, 224)
        label_tensor (torch.Tensor): True class labels (B,)
        epsilon (float): Attack magnitude in normalized space. Default: 0.03
                        (about 7.6 × 10^-3 per channel after std scaling)
    
    Returns:
        torch.Tensor: Adversarial image tensor with same shape as input
    
    Citation: Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015
    """
    # Clone input and enable gradient computation
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass: compute loss
    outputs = model(image_tensor)
    loss = F.cross_entropy(outputs, label_tensor)
    
    # Backward pass: compute gradients w.r.t. input
    loss.backward()
    
    # Extract gradients and compute perturbation in normalized space
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(1, 3, 1, 1)
    
    # Scale epsilon by std to match normalized image space
    eps_scaled = epsilon / std
    
    # Compute adversarial image: x + ε × sign(∇L)
    perturbed = image_tensor + eps_scaled * image_tensor.grad.sign()
    
    # Clamp to valid normalized image range
    # Original: [0, 1] → Normalized: [(0-mean)/std, (1-mean)/std]
    perturbed = torch.clamp(perturbed, (0 - mean) / std, (1 - mean) / std)
    
    return perturbed.detach()


# ============================
# Risk Stratification
# ============================
def get_risk_banner(pred_label: str) -> Tuple[str, str]:
    """
    Determine clinical risk level and display message based on prediction.
    
    Args:
        pred_label (str): Predicted class label ("NV" or "MEL")
    
    Returns:
        Tuple[str, str]:
            - message (str): Clinical recommendation
            - style (str): Streamlit alert type ("error", "success")
    """
    if pred_label == "MEL":
        return "🔴 MALIGNANT - High Risk: Immediate Consultation Recommended", "error"
    else:  # "NV"
        return "🟢 BENIGN - Low Risk: Routine Check Recommended", "success"


def smart_load_weights(model: nn.Module, base_dir: str) -> Tuple[bool, str]:
    """
    Intelligently load model weights from multiple candidate filenames.
    
    Tries these filenames in order:
    1. hare_final.pth (GA-optimized binary model)
    2. hare_best.pth (best checkpoint, may be 8-class)
    3. best_hare_model.pth (legacy filename)
    
    Handles multiple state dict formats:
    - 'model_state_dict' (saved with checkpoint dict)
    - 'state_dict' (alternative format)
    - Raw dict (direct model state)
    
    Uses strict=False to allow flexible loading (e.g., 8-class → 2-class with GA layers).
    
    Args:
        model (nn.Module): HARE_Ensemble model to load into
        base_dir (str): Directory to search for weight files
    
    Returns:
        Tuple[bool, str]:
            - success (bool): Whether weights were loaded successfully
            - message (str): Status message for UI display
    """
    candidates = ["hare_final.pth", "hare_best.pth", "best_hare_model.pth"]
    
    for filename in candidates:
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        
        try:
            state = torch.load(path, map_location="cpu")
            
            # Handle nested state dicts
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
            
            # Load with strict=False for flexibility
            model.load_state_dict(state, strict=False)
            return True, f"✅ Loaded binary weights from: {filename}"
        
        except Exception as exc:
            return False, f"❌ Failed to load {filename}: {exc}"
    
    message = f"⚠️ Binary weights not found. Expected one of: {', '.join(candidates)} in {base_dir}"
    return False, message


def get_device() -> torch.device:
    """Auto-detect GPU availability; fallback to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
