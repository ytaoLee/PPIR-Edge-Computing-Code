import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedCBAM(nn.Module):
    """Enhanced Convolutional Block Attention Module with improved components"""
    
    def __init__(self, channels, reduction_ratio=r, kernel_size=a, use_eca=False):
        super(EnhancedCBAM, self).__init__()
        self.use_eca = use_eca
        
        # Enhanced Channel Attention
        if use_eca:
            # Efficient Channel Attention (ECA) - lightweight alternative
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(x),
                nn.Conv1d(x, x, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
                nn.Sigmoid()
            )
        else:
            # Standard Channel Attention with bottleneck
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // reduction_ratio, kernel_size=x, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction_ratio, channels, kernel_size=x, bias=False),
                nn.Sigmoid()
            )
            
            # Add parallel max pooling branch for richer information
            self.max_pool_channel = nn.AdaptiveMaxPool2d(1)

        # Enhanced Spatial Attention with multi-scale context
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(a, b, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Residual connection scaling factor
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = x
        
        # Channel Attention
        if self.use_eca:
            # ECA implementation
            y = self.channel_attention(x)
        else:
            # Standard CBAM channel attention
            avg_out = self.channel_attention(x)
            max_out = self.channel_attention(self.max_pool_channel(x))
            y = avg_out + max_out
        
        x_channel = x * y.expand_as(x)
        
        # Spatial Attention with multi-scale context
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        
        # Apply spatial attention with residual connection
        x_att = x_channel * spatial_att.expand_as(x_channel)
        x_att = self.gamma * x_att + residual  # Residual connection
        
        return x_att


class AdvancedMultiScaleConvBlock(nn.Module):
    """Advanced multi-scale convolutional block with residual connections"""
    
    def __init__(self, in_channels=num1, out_channels=[dim1, dim2, dim3], 
                 use_residual=True, attention_reduction=g):
        super(AdvancedMultiScaleConvBlock, self).__init__()
        self.use_residual = use_residual
        
        # Multi-scale convolutional layers with increasing receptive fields
        self.conv1 = self._build_conv_block(in_channels, out_channels[0], kernel_size=x)
        self.conv2 = self._build_conv_block(out_channels[0], out_channels[1], kernel_size=x)
        self.conv3 = self._build_conv_block(out_channels[1], out_channels[2], kernel_size=x)
        
        # Additional parallel path with larger kernel for diverse features
        self.parallel_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0]//2, kernel_size=x2, padding=a),
            nn.BatchNorm2d(out_channels[0]//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        # Feature fusion for parallel path
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(out_channels[0] + out_channels[0]//2, out_channels[0], kernel_size=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        ) if use_residual else None
        
        # Enhanced attention mechanism
        self.cbam = EnhancedCBAM(out_channels[2], reduction_ratio=attention_reduction)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)

    def _build_conv_block(self, in_channels, out_channels, kernel_size=3):
        """Build a convolutional block with batch normalization and activation"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Main convolutional path
        x_main = self.conv1(x)
        x_main = self.conv2(x_main)
        x_main = self.conv3(x_main)
        
        # Parallel path with different receptive field
        if self.use_residual:
            x_parallel = self.parallel_conv(x)
            # Upsample parallel path to match main path size
            x_parallel = F.interpolate(x_parallel, size=x_main.shape[2:], 
                                     mode='bilinear', align_corners=False)
            # Fuse features
            x_fused = torch.cat([x_main, x_parallel], dim=1)
            x_main = self.feature_fusion(x_fused)
        
        # Apply attention and regularization
        x_att = self.cbam(x_main)
        x_att = self.dropout(x_att)
        
        return x_att


class EnhancedLocalFeatureExtractor(nn.Module):
    """Enhanced local feature extractor with multi-level feature aggregation"""
    
    def __init__(self, in_channels=b, feature_dim=dim3, use_multi_scale=True):
        super(EnhancedLocalFeatureExtractor, self).__init__()
        self.use_multi_scale = use_multi_scale
        
        # Multi-scale feature extraction
        self.multiscale_conv = AdvancedMultiScaleConvBlock(
            in_channels, [64, 128, 256], use_residual=True
        )
        
        # Additional convolutional layers for deeper features
        self.deep_layers = nn.Sequential(
            nn.Conv2d(dim2, dim3, kernel_size=a, padding=1),
            nn.BatchNorm2d(dim3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Multi-scale feature aggregation
        if use_multi_scale:
            self.scale_adapters = nn.ModuleList([
                nn.AdaptiveAvgPool2d(1),
                nn.AdaptiveAvgPool2d(1),
                nn.AdaptiveAvgPool2d(1)
            ])
            
            self.feature_fusion = nn.Sequential(
                nn.Linear(dim1 + dim2 + dim3 + dim4, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            )
        else:
            self.feature_proj = nn.Sequential(
                nn.Linear(512, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True)
            )
        
        # Feature normalization
        self.feature_norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        if self.use_multi_scale:
            # Extract features at different scales
            features = []
            
            # First scale
            x1 = self.multiscale_conv.conv1(x)
            features.append(self.scale_adapters[0](x1).flatten(1))
            
            # Second scale  
            x2 = self.multiscale_conv.conv2(x1)
            features.append(self.scale_adapters[1](x2).flatten(1))
            
            # Third scale
            x3 = self.multiscale_conv.conv3(x2)
            features.append(self.scale_adapters[2](x3).flatten(1))
            
            # Deep features
            x_deep = self.deep_layers(x3)
            features.append(x_deep.flatten(1))
            
            # Concatenate and fuse multi-scale features
            fused_features = torch.cat(features, dim=1)
            local_features = self.feature_fusion(fused_features)
        else:
            # Standard single-scale feature extraction
            features = self.multiscale_conv(x)
            deep_features = self.deep_layers(features)
            local_features = self.feature_proj(deep_features.flatten(1))
        
        # Normalize features
        local_features = self.feature_norm(local_features)
        
        return local_features
