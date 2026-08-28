"""Convolutional building blocks shared by the encoder and the decoder.

Unchanged from Descript Audio Codec (MIT licence).
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    """1-D convolution with weight normalisation."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """1-D transposed convolution with weight normalisation."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Periodic Snake activation with a learnable per-channel frequency."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)
