"""
Multimodal models for chest X-ray + radiology report classification.

Three fusion strategies are implemented:

  LateFusion   : image model and text model are trained independently,
                 their output logits are averaged at inference time.
                 Simple, robust, no information leakage across modalities.

  EarlyFusion  : image features and text CLS embedding are concatenated
                 before the classification head. The model jointly learns
                 a shared representation from the start.

  IntermediateFusion : both encoders produce intermediate representations
                 that are combined via a cross-attention layer before the head.
                 Richer interaction but more parameters.

Each model produces (B, num_classes) raw logits.
"""
import torch
import torch.nn as nn
from transformers import AutoModel
import timm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Image encoder (lightweight ViT from timm) ─────────────────────────────────

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0,
        )
        self.proj = nn.Linear(self.backbone.num_features, embed_dim)

    def forward(self, x):
        return self.proj(self.backbone(x))     # (B, embed_dim)


# ── Text encoder (Bio_ClinicalBERT) ───────────────────────────────────────────

class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = config.TEXT_MODEL_NAME,
        embed_dim: int = 512,
        freeze: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
        bert_dim = self.bert.config.hidden_size
        self.proj = nn.Linear(bert_dim, embed_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]       # [CLS] token: (B, bert_dim)
        return self.proj(cls)                   # (B, embed_dim)


# ── Late Fusion ───────────────────────────────────────────────────────────────

class LateFusionModel(nn.Module):
    """
    Trains image and text streams independently.
    Final prediction = average of both logit vectors.
    Handles missing modalities gracefully by using only the available one.
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES, embed_dim: int = 512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder  = TextEncoder(embed_dim=embed_dim)
        self.image_head    = nn.Linear(embed_dim, num_classes)
        self.text_head     = nn.Linear(embed_dim, num_classes)

    def forward(self, image=None, input_ids=None, attention_mask=None):
        logits = []
        if image is not None:
            img_feat  = self.image_encoder(image)
            logits.append(self.image_head(img_feat))
        if input_ids is not None:
            text_feat = self.text_encoder(input_ids, attention_mask)
            logits.append(self.text_head(text_feat))
        return torch.stack(logits, dim=0).mean(dim=0)


# ── Early Fusion ──────────────────────────────────────────────────────────────

class EarlyFusionModel(nn.Module):
    """
    Concatenates image and text embeddings before classification.
    Simple and fast; requires both modalities at inference.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        embed_dim: int = 512,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder  = TextEncoder(embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat  = self.image_encoder(image)
        text_feat = self.text_encoder(input_ids, attention_mask)
        fused     = torch.cat([img_feat, text_feat], dim=-1)
        return self.head(fused)


# ── Intermediate Fusion (Cross-Attention) ─────────────────────────────────────

class IntermediateFusionModel(nn.Module):
    """
    Image features attend over text features via a single cross-attention layer.
    Richer information exchange than early/late fusion.
    Requires both modalities at inference.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder  = TextEncoder(embed_dim=embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat  = self.image_encoder(image).unsqueeze(1)   # (B, 1, D)
        text_feat = self.text_encoder(input_ids, attention_mask).unsqueeze(1)  # (B, 1, D)

        # Image query attends to text key/value
        attended, _ = self.cross_attn(
            query=img_feat, key=text_feat, value=text_feat,
        )
        fused = self.norm(img_feat + attended).squeeze(1)   # (B, D)
        return self.head(fused)


# ── Text-only baseline ────────────────────────────────────────────────────────

class TextOnlyModel(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES, embed_dim: int = 512):
        super().__init__()
        self.encoder = TextEncoder(embed_dim=embed_dim)
        self.head    = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.head(self.encoder(input_ids, attention_mask))


# ── Image-only baseline ───────────────────────────────────────────────────────

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes: int = config.NUM_CLASSES, embed_dim: int = 512):
        super().__init__()
        self.encoder = ImageEncoder(embed_dim)
        self.head    = nn.Linear(embed_dim, num_classes)

    def forward(self, image, **kwargs):
        return self.head(self.encoder(image))
