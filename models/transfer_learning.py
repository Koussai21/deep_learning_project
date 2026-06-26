"""
Transfer learning models for multi-label chest pathology classification.

Three backbone options are provided:
  - DenseNet121  : strong baseline for chest X-rays (CheXNet heritage)
  - ResNet50     : standard residual network
  - EfficientNetB0 : efficient baseline with good accuracy/cost tradeoff

All models replace the original classification head with a linear layer
producing num_classes logits (sigmoid applied at inference / loss time).

Two training strategies:
  - frozen   : only the new head is trained (fast, few-shot style)
  - finetune : all weights updated with a lower LR for the backbone
"""
import torch
import torch.nn as nn
import torchvision.models as models
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def _build_densenet(num_classes: int, dropout: float, freeze_backbone: bool):
    backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    in_features = backbone.classifier.in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone


def _build_resnet(num_classes: int, dropout: float, freeze_backbone: bool):
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for name, p in backbone.named_parameters():
            if "layer4" not in name and "fc" not in name:
                p.requires_grad = False
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone


def _build_efficientnet(num_classes: int, dropout: float, freeze_backbone: bool):
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for p in backbone.features.parameters():
            p.requires_grad = False
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return backbone


BUILDERS = {
    "densenet121": _build_densenet,
    "resnet50":    _build_resnet,
    "efficientnet_b0": _build_efficientnet,
}


class TransferModel(nn.Module):
    """
    Wrapper around a pretrained torchvision backbone.
    Output: (B, num_classes) raw logits.
    """

    def __init__(
        self,
        backbone_name: str = "densenet121",
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        assert backbone_name in BUILDERS, f"Unknown backbone: {backbone_name}"
        self.backbone_name = backbone_name
        self.model = BUILDERS[backbone_name](num_classes, dropout, freeze_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
