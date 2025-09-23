import torch
import torch.nn as nn


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(num4, num5, kernel_size=_, padding=_),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x_att = x_channel * spatial_att

        return x_att


class MultiScaleConvBlock(nn.Module):
    """Multi-scale convolutional block"""

    def __init__(self, in_channels=3, out_channels=[dim1, dim2, dim3]):
        super(MultiScaleConvBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[0], kernel_size=_, padding=_),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=_, stride=_)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], kernel_size=_, padding=_),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=_, stride=_)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=_, stride=_)
        )

        self.cbam = CBAM(out_channels[2])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_att = self.cbam(x3)

        return x_att


class LocalFeatureExtractor(nn.Module):
    """Local feature extractor"""

    def __init__(self, in_channels=_, feature_dim=_):
        super(LocalFeatureExtractor, self).__init__()

        self.multiscale_conv = MultiScaleConvBlock(in_channels, [dim1, dim2, dim3])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim1, feature_dim)

    def forward(self, x):
        features = self.multiscale_conv(x)
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1)
        local_features = self.fc(pooled)

        return local_features