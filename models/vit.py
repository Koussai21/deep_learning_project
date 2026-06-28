import torch
import torch.nn as nn
import timm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ViTClassifier(nn.Module):

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        img_size: int = 128,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = config.DROPOUT,
        pretrained: bool = True,
    ):
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=dropout,
            img_size=img_size,
        )
        embed_dim = self.vit.embed_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))


class HybridCNNViT(nn.Module):

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        embed_dim: int = 192,
        num_heads: int = 6,
        num_layers: int = 2,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        import torchvision.models as tv_models

        resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2,
        )
        cnn_out_channels = 512

        self.proj = nn.Linear(cnn_out_channels, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

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

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        B, C, H, W = feat.shape
        tokens = feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.proj(tokens)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        out    = self.transformer(tokens)
        return self.head(self.norm(out[:, 0]))
