"""
Vision Transformer (ViT) and hybrid CNN/Transformer for multi-label classification.

Two options are provided:

1. ViTClassifier  - pure ViT from the timm library (vit_base_patch16_224).
   Images are split into non-overlapping 16×16 patches.
   Each patch is projected to an embedding dimension (768 for ViT-Base).
   A learnable [CLS] token aggregates global information via multi-head
   self-attention layers before the classification head.

2. HybridCNNViT  - CNN feature extractor (ResNet-style) followed by a
   Transformer encoder. The CNN produces a spatial feature map that is
   then treated as a sequence of tokens — combining local inductive biases
   of convolutions with the global context captured by attention.
"""
import torch
import torch.nn as nn
import timm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ViTClassifier(nn.Module):
    """
    Pure ViT (pretrained on ImageNet-21k via timm).
    Output: (B, num_classes) raw logits.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove default head
            drop_rate=dropout,
        )
        embed_dim = self.vit.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit(x)      # (B, embed_dim)
        return self.head(features)


class HybridCNNViT(nn.Module):
    """
    CNN encoder (first layers of ResNet50) + lightweight Transformer encoder.

    The CNN extracts local features and reduces spatial resolution.
    The resulting feature grid is flattened into a sequence of tokens,
    then processed by a 4-layer Transformer encoder.
    A [CLS] token aggregates the sequence and feeds the classification head.

    Output: (B, num_classes) raw logits.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        import torchvision.models as tv_models

        # ── CNN feature extractor (ResNet50 up to layer3) ──────────────────
        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3,
        )
        cnn_out_channels = 1024        # ResNet50 layer3 output channels

        # ── Linear projection to Transformer dimension ─────────────────────
        self.proj = nn.Linear(cnn_out_channels, embed_dim)

        # ── Learnable [CLS] token ──────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Transformer encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification head ────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN: (B, C_cnn, H', W')
        feat = self.cnn(x)
        B, C, H, W = feat.shape

        # Flatten spatial dims → sequence of tokens: (B, H*W, C_cnn)
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.proj(tokens)                    # (B, H*W, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, H*W+1, embed_dim)

        # Transformer
        out = self.transformer(tokens)                 # (B, H*W+1, embed_dim)
        cls_out = self.norm(out[:, 0])                 # take CLS token: (B, embed_dim)

        return self.head(cls_out)
