"""
models/sonar_net.py
===================
Multi-task CNN for spatial audio room parameter estimation.

Architecture:
  Input  : (batch, 1, 128, 128) log-Mel spectrogram
  Backbone: MobileNetV3-Small (pre-trained ImageNet weights, 1-ch adapted)
  Head 1 (Regression)    : [Width, Length, Height] in metres (3 outputs)
  Head 2 (Classification): Wall material class (5 outputs, softmax)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class SonarNetMobileV3(nn.Module):
    """
    MobileNetV3-Small backbone adapted for single-channel Mel-spectrogram input.
    
    The first conv layer weights are summed across the 3 input channels to
    produce a 1-channel equivalent (mean-adaptation).
    """

    def __init__(
        self,
        num_material_classes: int = 5,
        pretrained: bool = True,
        dim_output: int = 3,
    ) -> None:
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("Install torchvision: pip install torchvision")

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        # Adapt first conv: 3-ch → 1-ch by averaging weights
        orig_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            1, orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )
        with torch.no_grad():
            new_conv.weight.copy_(orig_conv.weight.sum(dim=1, keepdim=True) / 3.0)
            if orig_conv.bias is not None:
                new_conv.bias.copy_(orig_conv.bias)
        backbone.features[0][0] = new_conv

        # Remove the original classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Feature dimension from MobileNetV3-Small = 576
        feature_dim = 576

        # --- Dimension Regression Head ---
        self.dim_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, dim_output),          # [W, L, H]
            nn.Softplus(),                        # Ensure positive dimensions
        )

        # --- Material Classification Head ---
        self.mat_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(64, num_material_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, 128, 128)

        Returns
        -------
        dims : torch.Tensor
            Shape (B, 3) — predicted [Width, Length, Height] in metres.
        material_logits : torch.Tensor
            Shape (B, num_classes) — raw logits for material classification.
        """
        features = self.features(x)
        features = self.avgpool(features)
        features = features.flatten(1)
        dims = self.dim_head(features)
        material_logits = self.mat_head(features)
        return dims, material_logits


class SonarNetCustomCNN(nn.Module):
    """
    Ultra-lightweight custom CNN for environments where torchvision is not available.
    ~200K parameters, suitable for direct TFLite quantisation to <1 MB.
    """

    def __init__(
        self,
        num_material_classes: int = 5,
        dim_output: int = 3,
    ) -> None:
        super().__init__()

        def _dw_block(in_ch, out_ch, stride=1):
            """Depthwise separable conv block."""
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            _dw_block(16, 32, stride=2),
            _dw_block(32, 64, stride=2),
            _dw_block(64, 128, stride=2),
            _dw_block(128, 128, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.dim_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU6(inplace=True),
            nn.Linear(64, dim_output),
            nn.Softplus(),
        )
        self.mat_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU6(inplace=True),
            nn.Linear(32, num_material_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.dim_head(x), self.mat_head(x)


class SonarLoss(nn.Module):
    """
    Combined multi-task loss.

    L = w_dim * SmoothL1(dims) + w_mat * CrossEntropy(material)
    Supports optional class_weights for imbalanced material distributions.
    """

    def __init__(
        self,
        dim_weight: float = 1.0,
        mat_weight: float = 0.5,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.dim_weight = dim_weight
        self.mat_weight = mat_weight
        self.register_buffer("class_weights", class_weights)

    def forward(
        self,
        pred_dims: torch.Tensor,
        pred_mat: torch.Tensor,
        target_dims: torch.Tensor,
        target_mat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim_loss = F.smooth_l1_loss(pred_dims, target_dims)
        mat_loss = F.cross_entropy(pred_mat, target_mat, weight=self.class_weights)
        total = self.dim_weight * dim_loss + self.mat_weight * mat_loss
        return total, dim_loss, mat_loss


def build_model(
    backbone: str = "mobilenet_v3_small",
    num_material_classes: int = 5,
    pretrained: bool = True,
) -> nn.Module:
    """
    Model factory function.

    Parameters
    ----------
    backbone : str
        'mobilenet_v3_small' or 'custom_cnn'.
    num_material_classes : int
        Number of material classes (default 5).
    pretrained : bool
        Use ImageNet pre-trained weights (MobileNet only).

    Returns
    -------
    nn.Module
    """
    if backbone == "mobilenet_v3_small":
        return SonarNetMobileV3(num_material_classes=num_material_classes, pretrained=pretrained)
    elif backbone == "custom_cnn":
        return SonarNetCustomCNN(num_material_classes=num_material_classes)
    else:
        raise ValueError(f"Unknown backbone: '{backbone}'. Choose mobilenet_v3_small or custom_cnn.")


if __name__ == "__main__":
    model = SonarNetCustomCNN()
    x = torch.randn(2, 1, 128, 128)
    dims, mat = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Custom CNN | params: {n_params:,}")
    print(f"dims shape: {dims.shape}, mat shape: {mat.shape}")

