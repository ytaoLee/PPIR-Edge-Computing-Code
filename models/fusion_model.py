import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from ..backbones.vit import VisionTransformer
from ..backbones.resnet import ResNetBackbone

class MultiScaleFeatureExtractor(nn.Module):
    """Advanced multi-scale feature extractor with attention mechanisms"""
    
    def __init__(self, input_channels: int = a, feature_dims: List[int] = [dim1, dim2, dim3, dim4],
                 use_attention: bool = True, fusion_method: str = "concat"):
        super().__init__()
        self.feature_dims = feature_dims
        self.fusion_method = fusion_method
        self.use_attention = use_attention
        
        # Multi-scale convolutional layers
        self.conv_layers = nn.ModuleList()
        prev_channels = input_channels
        
        for dim in feature_dims:
            conv_block = self._create_conv_block(prev_channels, dim)
            self.conv_layers.append(conv_block)
            prev_channels = dim
        
        # Attention mechanisms
        if use_attention:
            self.attention_layers = nn.ModuleList([
                MultiHeadAttention(dim) for dim in feature_dims
            ])
        
        # Feature fusion
        if fusion_method == "concat":
            self.fusion_output_dim = sum(feature_dims)
        elif fusion_method == "weighted":
            self.fusion_weights = nn.Parameter(torch.ones(len(feature_dims)))
            self.fusion_output_dim = feature_dims[-1]
        
        self.output_projection = nn.Linear(self.fusion_output_dim, d)
    
    def _create_conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a convolutional block with batch norm and activation"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, a, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = []
        batch_size = x.shape[0]
        
        # Extract multi-scale features
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            
            # Apply attention if enabled
            if self.use_attention and hasattr(self, 'attention_layers'):
                x = self.attention_layers[i](x)
            
            # Global average pooling for feature vector
            pooled = F.adaptive_avg_pool2d(x, 1).view(batch_size, -1)
            features.append(pooled)
        
        # Fuse features
        if self.fusion_method == "concat":
            fused = torch.cat(features, dim=1)
        elif self.fusion_method == "weighted":
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, features))
        else:
            fused = features[-1]  # Use last layer features
        
        return self.output_projection(fused)

class AdvancedFeatureFusion(nn.Module):
    """Advanced feature fusion with transformer-based integration"""
    
    def __init__(self, global_feat_dim: int = a, local_feat_dim: int = a,
                 num_heads: int = b, num_layers: int = c, hidden_dim: int = dim3):
        super().__init__()
        
        # Feature projection layers
        self.global_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.local_proj = nn.Sequential(
            nn.Linear(local_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Cross-attention fusion
        self.cross_attention = CrossModalAttention(
            hidden_dim, num_heads, dropout=0.x
        )
        
        # Transformer encoder for deep fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.x,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.x),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh()
        )
    
    def forward(self, global_features: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        # Project features to common dimension
        global_proj = self.global_proj(global_features)
        local_proj = self.local_proj(local_features)
        
        # Apply cross-attention
        attended_features = self.cross_attention(global_proj, local_proj)
        
        # Further processing with transformer
        fused_features = self.transformer_encoder(attended_features.unsqueeze(1))
        fused_features = fused_features.squeeze(1)
        
        return self.output_proj(fused_features)

class CrossModalAttention(nn.Module):
    """Cross-modal attention for feature fusion"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.x):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        # Query: global features, Key/Value: local features
        attended, _ = self.multihead_attn(
            query.unsqueeze(1),  # Add sequence dimension
            key_value.unsqueeze(1),
            key_value.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(attended))
        return output
