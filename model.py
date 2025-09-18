import torch
import torch.nn as nn
from torchvision import models

class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: B x C
        y = self.fc(x)
        return x * y

class ResNetFeature(nn.Module):
    def __init__(self, model_name='resnet18'):
        super(ResNetFeature, self).__init__()
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=False)
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=False)
        else:
            raise ValueError("Unsupported ResNet model")
        self.features = nn.Sequential(
            *list(base_model.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x):
        return self.features(x)

class ViTFeature(nn.Module):
    def __init__(self):
        super(ViTFeature, self).__init__()
        base_model = models.vit_b_16(pretrained=False)
        self.features = base_model.forward_features
        self.proj = nn.Linear(768, 512)

    def forward(self, x):
        feats = self.features(x)
        cls_token = feats[:, 0]  # CLS token: B x 768
        return self.proj(cls_token)

class MultiModalModel(nn.Module):
    def __init__(self, embed_dim=512):
        super(MultiModalModel, self).__init__()
        self.embed_dim = embed_dim

        # Image backbones
        self.i1_backbone = ResNetFeature('resnet34')  # outputs B x 512
        self.i2_backbone = ViTFeature()  # outputs B x 512 (after proj)
        self.i3_backbone = ResNetFeature('resnet18')  # outputs B x 512

        # SE-Attentions for image branch
        self.se1 = SEAttention(512)
        self.se2 = SEAttention(1024)  # 512 + 512
        self.se3 = SEAttention(1536)  # 1024 + 512

        # v branch
        self.v_dense = nn.Linear(9, embed_dim)  # vd: B x 512
        self.v_embed = nn.Linear(1, embed_dim)  # for self-attn
        self.v_self_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.v_pool = nn.AdaptiveAvgPool1d(1)  # not needed, use mean

        # Projections for cross-attention
        self.i_proj = nn.Linear(3072, embed_dim)  # 1536 + 512 + 1024 = 3072
        self.v_output_proj = nn.Linear(2 * embed_dim, embed_dim)  # 512 + 512 = 1024

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, i1, i2, i3, v):
        # Image processing
        i1r = self.i1_backbone(i1)  # B x 512
        i2v = self.i2_backbone(i2)  # B x 512
        i3r = self.i3_backbone(i3)  # B x 512

        i1rs = self.se1(i1r)  # B x 512
        i1rs_i2v = torch.cat([i1rs, i2v], dim=1)  # B x 1024
        i1rs_i2vs = self.se2(i1rs_i2v)  # B x 1024
        i1rs_i2vs_i3r = torch.cat([i1rs_i2vs, i3r], dim=1)  # B x 1536
        i1rs_i2vs_i3rs = self.se3(i1rs_i2vs_i3r)  # B x 1536

        i_output = torch.cat([i1rs_i2vs_i3rs, i1r, i1rs_i2v], dim=1)  # B x 3072

        # v processing
        vd = self.v_dense(v)  # B x 512

        # Self-attention on v: treat as sequence of length 9
        v_seq = v.unsqueeze(-1)  # B x 9 x 1
        v_embed = self.v_embed(v_seq)  # B x 9 x 512
        vs_attn, _ = self.v_self_attn(v_embed, v_embed, v_embed)  # B x 9 x 512
        vs = vs_attn.mean(dim=1)  # B x 512 (mean pool)

        v_output = torch.cat([vd, vs], dim=1)  # B x 1024

        # Cross-attention: i as query, v as key/value
        i_q = self.i_proj(i_output).unsqueeze(1)  # B x 1 x 512
        v_kv = self.v_output_proj(v_output).unsqueeze(1)  # B x 1 x 512
        total_out, _ = self.cross_attn(i_q, v_kv, v_kv)
        total_out = total_out.squeeze(1)  # B x 512

        return total_out

