"""
HARE_Ensemble: Hybrid Attention ResNet Ensemble for Skin Lesion Classification

This module defines the HARE_Ensemble model architecture, which combines:
- ResNet50 (CNN) for texture and local feature extraction (2048-dim)
- Vision Transformer (ViT) for global structure understanding (768-dim)
- Fusion layer optimized via Genetic Algorithm for adversarial robustness

Binary Classification: Class 0 = Nevus (Benign), Class 1 = Melanoma (Malignant)

Citation:
  HARE: Hybrid Adversarially-Robust Ensemble for Skin Cancer Detection (FYP)
  Uses OATGA (Optimization-Augmented Training with Genetic Algorithm) for fusion optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class HARE_Ensemble(nn.Module):
    """
    Hybrid Attention ResNet Ensemble for binary skin lesion classification.
    
    Architecture:
    - ResNet50 (ImageNet1K_V2 pretrained) → 2048-dim features
    - ViT Base-16/224 (ImageNet pretrained) → 768-dim features
    - Fusion Layer (FC 2816 → 512, evolved by GA) → ReLU + Dropout
    - Classifier (FC 512 → 2 classes)
    
    Args:
        num_classes (int): Number of output classes. Default: 2 (Nevus, Melanoma)
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # ============ ResNet50 Backbone ============
        # Pretrained on ImageNet1K_V2 for texture/local feature extraction
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final fully-connected layer; use intermediate features
        self.cnn.fc = nn.Identity()
        
        # ============ Vision Transformer Backbone ============
        # Pretrained on ImageNet for global structure understanding
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0  # Extract features, no classification head
        )
        
        # ============ Fusion & Classification ============
        # Fusion layer: concatenate CNN (2048) + ViT (768) features → 512 latent
        # This layer is the "novelty" of HARE: optimized via Genetic Algorithm (OATGA)
        # for balanced accuracy (30%) and adversarial robustness (70%) trade-off
        self.fusion_fc = nn.Linear(2048 + 768, 512)
        
        # Final classifier: 512-dim latent → 2 classes
        self.classifier = nn.Linear(512, num_classes)
        
        # Dropout for regularization (applied after fusion)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Fuse CNN and ViT features, then classify.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, 3, 224, 224)
        
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        # Extract CNN features (B, 2048)
        f_cnn = self.cnn(x)
        
        # Extract ViT features (B, 768)
        f_vit = self.vit(x)
        
        # Concatenate features along channel dimension (B, 2816)
        combined = torch.cat((f_cnn, f_vit), dim=1)
        
        # Fuse features through GA-optimized fusion layer + ReLU activation
        x = F.relu(self.fusion_fc(combined))
        
        # Apply dropout for robustness
        x = self.dropout(x)
        
        # Classify: fusion features → logits (B, 2)
        return self.classifier(x)
