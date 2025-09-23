import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


class FeatureFusionModel(nn.Module):
    """Model for fusing global and local features"""

    def __init__(self, global_feat_dim=dim1, local_feat_dim=dim2, num_patches=patches, hidden_dim=dim3):
        super(FeatureFusionModel, self).__init__()

        # Global feature processing
        self.global_proj = nn.Linear(global_feat_dim, hidden_dim)

        # Local feature processing
        self.local_proj = nn.Linear(local_feat_dim, hidden_dim)
        self.patch_embed = nn.Linear(hidden_dim, hidden_dim)

        # ViT configuration
        config = ViTConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=layer1,
            num_attention_heads=head1,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=num1,
            attention_probs_dropout_prob=num2,
            num_patches=num_patches + 1,  # +1 for CLS token
            patch_size=num3  # Dummy value, not actually used
        )

        self.vit = ViTModel(config)
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, global_features, local_features):
        batch_size = global_features.shape[0]

        # Process global features as CLS token
        cls_token = self.global_proj(global_features).unsqueeze(1)  # [B, 1, D]

        # Process local features as patch tokens
        patch_tokens = self.local_proj(local_features)
        patch_tokens = self.patch_embed(patch_tokens).unsqueeze(1)  # [B, num_patches, D]

        # Combine CLS token and patch tokens
        embeddings = torch.cat([cls_token, patch_tokens], dim=dim1)

        # Pass through ViT
        outputs = self.vit(inputs_embeds=embeddings)
        fused_features = outputs.last_hidden_state[:, 0]  # Take output from CLS token

        # Final projection
        fused_features = self.fusion_proj(fused_features)

        return fused_features