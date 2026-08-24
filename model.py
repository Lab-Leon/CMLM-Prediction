import torch
import torch.nn as nn
from torchvision import models


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ResNetFeature(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        if model_name == "resnet18":
            backbone = models.resnet18(weights=None)
        elif model_name == "resnet34":
            backbone = models.resnet34(weights=None)
        else:
            raise ValueError("model_name must be 'resnet18' or 'resnet34'")
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


class ViTFeature(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        backbone = models.vit_b_16(weights=None)
        backbone.heads = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(768, embed_dim)

    def forward(self, x):
        return self.projection(self.backbone(x))


class MultiModalModel(nn.Module):
    def __init__(self, embed_dim=512, num_numerical_features=9, num_classes=2):
        super().__init__()
        self.num_numerical_features = num_numerical_features

        # Multi-scale image feature extraction (MSEM principle).
        self.original_backbone = ResNetFeature("resnet34")
        self.roi_position_backbone = ViTFeature(embed_dim)
        self.roi_backbone = ResNetFeature("resnet18")

        self.se_original = SEAttention(embed_dim)
        self.se_pair = SEAttention(2 * embed_dim)
        self.se_all = SEAttention(3 * embed_dim)
        self.image_projection = nn.Linear(3 * embed_dim, embed_dim)
        self.image_token_embedding = nn.Parameter(torch.zeros(1, 3, embed_dim))

        # Numerical feature extraction (SADM principle).
        self.numerical_dense = nn.Sequential(
            nn.Linear(num_numerical_features, embed_dim),
            nn.ReLU(inplace=True),
        )
        self.numerical_scalar_embedding = nn.Linear(1, embed_dim)
        self.numerical_feature_embedding = nn.Parameter(
            torch.zeros(1, num_numerical_features, embed_dim)
        )
        self.numerical_self_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.numerical_norm = nn.LayerNorm(embed_dim)
        self.numerical_projection = nn.Linear(2 * embed_dim, embed_dim)

        # Image tokens attend over all nine numerical tokens. Unlike a
        # one-key attention layer, this performs a genuine cross-modal lookup.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_classes),
        )

        nn.init.normal_(self.image_token_embedding, std=0.02)
        nn.init.normal_(self.numerical_feature_embedding, std=0.02)

    def forward_features(
        self, original_image, roi_with_position, roi_without_position, numerical
    ):
        if numerical.ndim != 2 or numerical.size(1) != self.num_numerical_features:
            raise ValueError(
                f"numerical must have shape [batch, {self.num_numerical_features}]"
            )

        original = self.original_backbone(original_image)
        roi_position = self.roi_position_backbone(roi_with_position)
        roi = self.roi_backbone(roi_without_position)

        original_se = self.se_original(original)
        image_pair = self.se_pair(torch.cat([original_se, roi_position], dim=1))
        image_all = self.se_all(torch.cat([image_pair, roi], dim=1))
        image_global = self.image_projection(image_all)

        image_tokens = torch.stack([original, roi_position, roi], dim=1)
        image_tokens = image_tokens + self.image_token_embedding

        numerical_dense = self.numerical_dense(numerical)
        numerical_tokens = self.numerical_scalar_embedding(numerical.unsqueeze(-1))
        numerical_tokens = numerical_tokens + self.numerical_feature_embedding
        numerical_attended, _ = self.numerical_self_attention(
            numerical_tokens, numerical_tokens, numerical_tokens, need_weights=False
        )
        numerical_tokens = self.numerical_norm(
            numerical_tokens + numerical_attended
        )
        numerical_global = self.numerical_projection(
            torch.cat([numerical_dense, numerical_tokens.mean(dim=1)], dim=1)
        )

        cross_attended, _ = self.cross_attention(
            image_tokens, numerical_tokens, numerical_tokens, need_weights=False
        )
        cross_global = self.cross_norm(image_tokens + cross_attended).mean(dim=1)

        return torch.cat([image_global, numerical_global, cross_global], dim=1)

    def forward(
        self, original_image, roi_with_position, roi_without_position, numerical
    ):
        fused = self.forward_features(
            original_image, roi_with_position, roi_without_position, numerical
        )
        return self.classifier(fused)
