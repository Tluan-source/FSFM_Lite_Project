import torch
import torch.nn as nn


class SpatialConsistency(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


class FeatureConsistency(nn.Module):

    def __init__(self, embed_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        mlp_out = self.mlp(x)
        return self.norm(x + mlp_out)


class SemanticConsistency(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patches, cls_token):
        cls_token = cls_token.unsqueeze(1)
        cls_expand = cls_token.expand(-1, patches.shape[1], -1)
        attn_out, _ = self.attn(patches, cls_expand, cls_expand)

        return self.norm(patches + attn_out)


class FusionBlock(nn.Module):

    def __init__(self, embed_dim=768):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, spatial_feat, feature_feat, semantic_feat):

        fusion = torch.cat(
            [
                spatial_feat,
                feature_feat,
                semantic_feat
            ],
            dim=-1
        )

        return self.proj(fusion)


class ThreeCModule(nn.Module):

    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.spatial = SpatialConsistency(embed_dim, num_heads)
        self.feature = FeatureConsistency(embed_dim)
        self.semantic = SemanticConsistency(embed_dim, num_heads)
        self.fusion = FusionBlock(embed_dim)

    def forward(self, patch_tokens, cls_token):

        spatial_feat = self.spatial(patch_tokens)
        feature_feat = self.feature(patch_tokens)
        semantic_feat = self.semantic(patch_tokens, cls_token)
        enhanced_tokens = self.fusion(spatial_feat, feature_feat, semantic_feat)
        return enhanced_tokens