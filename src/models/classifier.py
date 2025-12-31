"""OCT Classification model using timm backbones."""

from typing import Optional

import timm
import torch
import torch.nn as nn


class OCTClassifier(nn.Module):
    """Retinal OCT Classifier using pretrained backbones from timm.

    Supports various modern architectures:
    - EfficientNet (efficientnet_b0 to efficientnet_b7)
    - ConvNeXt (convnext_tiny, convnext_small, convnext_base)
    - Vision Transformer (vit_small_patch16_224, vit_base_patch16_224)
    - ResNet (resnet50, resnet101)
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """Initialize the classifier.

        Args:
            backbone: Name of the timm backbone model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability before final layer
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Create backbone with timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="",  # Remove global pooling
        )

        # Get the number of features from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:  # CNN output: (B, C, H, W)
                num_features = features.shape[1]
            else:  # Transformer output: (B, N, C)
                num_features = features.shape[-1]

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1) if len(features.shape) == 4 else None

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self):
        """Initialize the classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)

        if self.global_pool is not None:
            features = self.global_pool(features)
        else:
            # For transformers, take CLS token or mean pool
            if len(features.shape) == 3:
                features = features.mean(dim=1)

        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before the classifier head.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Features tensor
        """
        features = self.backbone(x)
        if self.global_pool is not None:
            features = self.global_pool(features)
        return features

    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the backbone parameters.

        Args:
            freeze: Whether to freeze backbone
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def unfreeze_last_n_layers(self, n: int = 2):
        """Unfreeze the last n layers of the backbone for fine-tuning.

        Args:
            n: Number of layers to unfreeze from the end
        """
        # First freeze everything
        self.freeze_backbone(True)

        # Get all named parameters
        params = list(self.backbone.named_parameters())

        # Unfreeze last n layers (approximate by looking at parameter names)
        layers_seen = set()
        for name, param in reversed(params):
            layer_name = name.split(".")[0]
            if layer_name not in layers_seen:
                layers_seen.add(layer_name)
            if len(layers_seen) <= n:
                param.requires_grad = True


def create_model(config: dict) -> OCTClassifier:
    """Create model from config.

    Args:
        config: Configuration dictionary

    Returns:
        OCTClassifier instance
    """
    model_config = config.get("model", {})

    return OCTClassifier(
        backbone=model_config.get("backbone", "efficientnet_b3"),
        num_classes=model_config.get("num_classes", 4),
        pretrained=model_config.get("pretrained", True),
        dropout=model_config.get("dropout", 0.3),
    )
