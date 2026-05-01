"""
Thesis-aligned HARE model definition.

This runtime mirrors the architecture described in `HARE_Thesis_Draft.md` and
the notebooks in `model-development`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
import timm


class HAREThesisModel(nn.Module):
    """Hybrid ResNet-50 + ViT-Small model with multi-head outputs."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.35,
        pretrained_backbones: bool = False,
        use_vit_grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        cnn_weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbones else None
        self.cnn = models.resnet50(weights=cnn_weights)
        cnn_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.vit = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained_backbones,
            num_classes=0,
        )
        if use_vit_grad_checkpoint and hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(True)
        vit_dim = self.vit.num_features

        self.cnn_head = nn.Linear(cnn_dim, num_classes)
        self.vit_head = nn.Linear(vit_dim, num_classes)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        fused_feat = self.fusion(torch.cat([cnn_feat, vit_feat], dim=1))
        return {
            "cnn_features": cnn_feat,
            "vit_features": vit_feat,
            "cnn_logits": self.cnn_head(cnn_feat),
            "vit_logits": self.vit_head(vit_feat),
            "fusion_logits": self.fusion_head(fused_feat),
        }


HARE_Ensemble = HAREThesisModel
