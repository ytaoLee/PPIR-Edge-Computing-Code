import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct


class DCTGlobalFeatureExtractor(nn.Module):
    def __init__(self, num_bins=num1, feature_dim=dim1):
        super(DCTGlobalFeatureExtractor, self).__init__()
        self.num_bins = num_bins

        # DCT coefficient histogram processing network
        self.fc1 = nn.Linear(num_bins, num2)
        self.ln1 = nn.LayerNorm(num3)
        self.fc2 = nn.Linear(num4, feature_dim)
        self.relu = nn.ReLU()

    def compute_dct_histogram(self, x):
        """Compute DCT coefficient histogram"""
        batch_size, channels, height, width = x.shape

        # Convert to numpy for DCT computation
        x_np = x.cpu().numpy()
        dct_features = []

        for i in range(batch_size):
            for c in range(channels):
                # Compute 2D DCT
                dct_coeffs = dct(dct(x_np[i, c], axis=0, norm='ortho'), axis=1, norm='ortho')

                # Take absolute values and flatten
                dct_mag = np.abs(dct_coeffs).flatten()

                # Compute histogram
                hist, _ = np.histogram(dct_mag, bins=self.num_bins, range=(0, np.max(dct_mag)))
                hist = hist / (height * width)  # Normalization

                dct_features.append(hist)

        return torch.tensor(np.array(dct_features), device=x.device, dtype=x.dtype)

    def forward(self, x):
        # Compute DCT histogram
        dct_hist = self.compute_dct_histogram(x)

        # Pass through fully connected network
        x = self.relu(self.ln1(self.fc1(dct_hist)))
        global_feature = self.fc2(x)

        return global_feature