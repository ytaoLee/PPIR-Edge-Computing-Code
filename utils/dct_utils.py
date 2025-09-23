import numpy as np
from scipy.fftpack import dct, idct


def dct2(block):
    """Compute 2D Discrete Cosine Transform"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def idct2(block):
    """Compute 2D Inverse Discrete Cosine Transform"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def compute_dct_histogram(image, block_size=8, num_bins=64):
    """
    Compute DCT coefficient histogram for an image

    Args:
        image: Input image (H, W, C) or (H, W)
        block_size: DCT block size
        num_bins: Number of histogram bins

    Returns:
        hist: Normalized histogram feature vector
    """
    if len(image.shape) == 3:
        # Convert to grayscale
        image = np.mean(image, axis=2)

    h, w = image.shape
    hist = np.zeros(num_bins)

    # Compute DCT block by block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            if block.shape[0] == block_size and block.shape[1] == block_size:
                dct_block = dct2(block)
                # Take absolute values and exclude DC coefficient
                dct_mag = np.abs(dct_block)
                dct_mag[0, 0] = 0  # Exclude DC coefficient

                # Compute histogram
                flat_dct = dct_mag.flatten()
                hist_block, _ = np.histogram(flat_dct, bins=num_bins, range=(0, np.max(flat_dct)))
                hist += hist_block

    # Normalize histogram
    if np.sum(hist) > 0:
        hist = hist / np.sum(hist)

    return hist