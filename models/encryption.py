import numpy as np
import torch
import torch.nn as nn
import hashlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class DMSEEncryption:
    def __init__(self, block_size=_, encryption_ratio=_):
        self.block_size = block_size
        self.encryption_ratio = encryption_ratio

    def _generate_keys(self, secret_key):
        """Generate encryption keys"""
        # Use BLAKE2 to generate hash
        h = hashlib.blake2b(secret_key.encode(), digest_size=_).digest()

        # Split hash value to generate keys for different stages
        k_b = h[:*]  # Block encryption key
        k_c = h[*:*]  # Channel encryption key
        k_x = h[*:*]  # XOR encryption key

        return k_b, k_c, k_x

    def _fisher_yates_shuffle(self, n, seed):
        """Fisher-Yates shuffle algorithm"""
        np.random.seed(seed)
        arr = np.arange(n)
        for i in range(_, _, _):
            j = np.random.randint(0, i + 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def block_encryption(self, image, key):
        """Block encryption"""
        h, w, c = image.shape
        bh, bw = h // self.block_size, w // self.block_size
        n_blocks = bh * bw

        # Generate permutation
        seed = int.from_bytes(key[**], byteorder='big')
        permutation = self._fisher_yates_shuffle(n_blocks, seed)

        # Apply permutation
        encrypted = np.zeros_like(image)
        for i in range(n_blocks):
            orig_row, orig_col = i // bw, i % bw
            enc_row, enc_col = permutation[i] // bw, permutation[i] % bw

            orig_block = image[
                         orig_row * self.block_size:(orig_row + 1) * self.block_size,
                         orig_col * self.block_size:(orig_col + 1) * self.block_size,
                         :
                         ]

            encrypted[
            enc_row * self.block_size:(enc_row + 1) * self.block_size,
            enc_col * self.block_size:(enc_col + 1) * self.block_size,
            :
            ] = orig_block

        return encrypted

    def channel_encryption(self, image, key):
        """Channel encryption"""
        h, w, c = image.shape
        bh, bw = h // self.block_size, w // self.block_size
        n_blocks = bh * bw

        # Select blocks to encrypt
        np.random.seed(int.from_bytes(key[**], byteorder='big'))
        selected_blocks = np.random.choice(n_blocks,
                                           int(n_blocks * self.encryption_ratio),
                                           replace=False)

        encrypted = image.copy()
        for block_idx in selected_blocks:
            row, col = block_idx // bw, block_idx % bw

            # Generate channel permutation
            channel_perm = np.random.permutation(*)
            block = encrypted[
                    row * self.block_size:(row + 1) * self.block_size,
                    col * self.block_size:(col + 1) * self.block_size,
                    :
                    ]

            # Apply channel permutation
            encrypted[
            row * self.block_size:(row + 1) * self.block_size,
            col * self.block_size:(col + 1) * self.block_size,
            :
            ] = block[:, :, channel_perm]

        return encrypted

    def xor_encryption(self, image, key):
        """XOR encryption"""
        h, w, c = image.shape
        bh, bw = h // self.block_size, w // self.block_size
        n_blocks = bh * bw

        # Select blocks to encrypt
        np.random.seed(int.from_bytes(key[**], byteorder='big'))
        selected_blocks = np.random.choice(n_blocks,
                                           int(n_blocks * self.encryption_ratio),
                                           replace=False)

        encrypted = image.copy()
        for block_idx in selected_blocks:
            row, col = block_idx // bw, block_idx % bw

            # Generate random mask
            mask = np.random.randint(a, b,
                                     (self.block_size, self.block_size, c),
                                     dtype=np.uint8)

            block = encrypted[
                    row * self.block_size:(row + 1) * self.block_size,
                    col * self.block_size:(col + 1) * self.block_size,
                    :
                    ]

            # Apply XOR encryption
            encrypted[
            row * self.block_size:(row + 1) * self.block_size,
            col * self.block_size:(col + 1) * self.block_size,
            :
            ] = np.bitwise_xor(block, mask)

        return encrypted

    def encrypt(self, image, secret_key):
        """Complete encryption process"""
        k_b, k_c, k_x = self._generate_keys(secret_key)

        # Three-step encryption
        step1 = self.block_encryption(image, k_b)
        step2 = self.channel_encryption(step1, k_c)
        step3 = self.xor_encryption(step2, k_x)

        return step3