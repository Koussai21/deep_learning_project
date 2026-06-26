"""
CNN from scratch for multi-label chest pathology classification.

Architecture:
  4 convolutional blocks (Conv → BN → ReLU → MaxPool),
  each doubling the number of channels (32 → 64 → 128 → 256).
  Global average pooling collapses spatial dims.
  Two fully-connected layers with Dropout lead to 14 sigmoid outputs.
"""
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class CNNFromScratch(nn.Module):
    """
    Simple CNN baseline trained end-to-end without pretrained weights.
    Input : (B, 3, image_size, image_size)
    Output: (B, num_classes)  — raw logits, apply sigmoid for probabilities.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  dropout=0.1),
            ConvBlock(32,  64,  dropout=0.1),
            ConvBlock(64,  128, dropout=0.2),
            ConvBlock(128, 256, dropout=0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the feature map before global pooling (for CAM visualisation)."""
        return self.features(x)
