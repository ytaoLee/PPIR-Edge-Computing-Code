import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DCTProcessor(nn.Module):
    """DCT processing module that works directly on GPU tensors"""
    
    def __init__(self, block_size=8, num_bins=64):
        super(DCTProcessor, self).__init__()
        self.block_size = block_size
        self.num_bins = num_bins
        
        # Precompute DCT basis for efficiency
        self.register_buffer('dct_basis', self._create_dct_basis(block_size))
        
    def _create_dct_basis(self, size):
        """Create DCT basis matrix"""
        basis = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if i == 0:
                    basis[i, j] = np.sqrt(1/size) * np.cos(np.pi * i * (2*j + 1) / (2*size))
                else:
                    basis[i, j] = np.sqrt(2/size) * np.cos(np.pi * i * (2*j + 1) / (2*size))
        return basis

    def dct2(self, x):
        """2D DCT using precomputed basis"""
        # x shape: [batch, channels, height, width]
        batch_size, channels, h, w = x.shape
        
        # Ensure divisible by block size
        h_pad = ((h + self.block_size - 1) // self.block_size) * self.block_size
        w_pad = ((w + self.block_size - 1) // self.block_size) * self.block_size
        
        if h_pad != h or w_pad != w:
            x = F.pad(x, (0, w_pad - w, 0, h_pad - h))
        
        # Reshape to blocks
        x_blocks = x.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        x_blocks = x_blocks.contiguous().view(batch_size, channels, -1, self.block_size, self.block_size)
        
        # Apply DCT to each block
        dct_blocks = torch.einsum('ij,bcnjk,kl->bcnil', self.dct_basis, x_blocks, self.dct_basis)
        
        return dct_blocks

    def compute_dct_histogram(self, x, exclude_dc=True):
        """Compute DCT histogram directly on GPU"""
        batch_size, channels, h, w = x.shape
        
        # Apply 2D DCT
        dct_coeffs = self.dct2(x)  # [batch, channels, num_blocks, block_size, block_size]
        
        # Flatten DCT coefficients
        dct_flat = dct_coeffs.view(batch_size, channels, -1, self.block_size * self.block_size)
        
        # Exclude DC coefficient if requested
        if exclude_dc:
            dct_flat = dct_flat[:, :, :, 1:]
        
        # Take absolute values
        dct_mag = torch.abs(dct_flat)
        
        # Compute histogram using torch.histc (approximate)
        hist_min = 0
        hist_max = torch.max(dct_mag) * 1.1  # Add small margin
        
        # Compute histogram for each block and channel
        hists = []
        for b in range(batch_size):
            batch_hists = []
            for c in range(channels):
                channel_hist = torch.histc(dct_mag[b, c], bins=self.num_bins, 
                                         min=hist_min, max=hist_max)
                batch_hists.append(channel_hist)
            hists.append(torch.stack(batch_hists))
        
        hists = torch.stack(hists)
        
        # Normalize histograms
        hists = hists / (h * w)  # Normalize by image size
        
        return hists.view(batch_size, -1)  # Flatten channel histograms


class EnhancedDCTGlobalFeatureExtractor(nn.Module):
    """Enhanced DCT global feature extractor with improved architecture"""
    
    def __init__(self, num_bins=64, feature_dim=256, use_multi_scale_dct=True):
        super(EnhancedDCTGlobalFeatureExtractor, self).__init__()
        self.num_bins = num_bins
        self.use_multi_scale_dct = use_multi_scale_dct
        
        # Multi-scale DCT processing
        if use_multi_scale_dct:
            self.dct_processors = nn.ModuleList([
                DCTProcessor(block_size=4, num_bins=num_bins//2),
                DCTProcessor(block_size=8, num_bins=num_bins),
                DCTProcessor(block_size=16, num_bins=num_bins//2)
            ])
            total_bins = (num_bins//2 + num_bins + num_bins//2) * 3  # 3 scales × 3 channels
        else:
            self.dct_processor = DCTProcessor(block_size=8, num_bins=num_bins)
            total_bins = num_bins * 3  # 3 channels
        
        # Enhanced feature processing network
        self.feature_network = nn.Sequential(
            nn.Linear(total_bins, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 384),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(384, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Tanh()  # Normalize outputs to [-1, 1]
        )
        
        # Attention mechanism for feature refinement
        self.feature_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(feature_dim)

    def extract_multi_scale_dct_features(self, x):
        """Extract DCT features at multiple scales"""
        multi_scale_features = []
        
        for processor in self.dct_processors:
            # Compute DCT histogram for each scale
            dct_hist = processor.compute_dct_histogram(x)
            multi_scale_features.append(dct_hist)
        
        # Concatenate multi-scale features
        return torch.cat(multi_scale_features, dim=1)

    def forward(self, x):
        # Extract DCT features
        if self.use_multi_scale_dct:
            dct_features = self.extract_multi_scale_dct_features(x)
        else:
            dct_features = self.dct_processor.compute_dct_histogram(x)
        
        # Process through feature network
        global_features = self.feature_network(dct_features)
        
        # Apply feature attention
        attention_weights = self.feature_attention(global_features)
        refined_features = global_features * attention_weights
        
        # Normalize features
        normalized_features = self.layer_norm(refined_features)
        
        return normalized_features

    def get_feature_importance(self, x):
        """Get feature importance scores for interpretability"""
        with torch.no_grad():
            if self.use_multi_scale_dct:
                dct_features = self.extract_multi_scale_dct_features(x)
            else:
                dct_features = self.dct_processor.compute_dct_histogram(x)
            
            # Get intermediate activations
            features = []
            current = dct_features
            
            for layer in self.feature_network:
                current = layer(current)
                if isinstance(layer, nn.ReLU):
                    features.append(current.cpu().numpy())
            
            return global_features
