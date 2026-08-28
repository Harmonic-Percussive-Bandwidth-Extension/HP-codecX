"""Multi-period and multi-band multi-scale STFT discriminators.

Unchanged from Descript Audio Codec (MIT licence), except that ``Discriminator``
holds one full set of sub-discriminators per frequency branch and dispatches on
the sampling rate of the signal it is given.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import STFTParams
from audiotools import ml
from einops import rearrange
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv1d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Module):
    """Multi-period discriminator: reshapes the waveform by a fixed period."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False
        )

    def pad_to_period(self, x):
        t = x.shape[-1]
        return F.pad(x, (0, self.period - t % self.period), mode="reflect")

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class MSD(nn.Module):
    """Multi-scale discriminator on the resampled waveform."""

    def __init__(self, rate: int = 1, sample_rate: int = 44100):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv1d(1, 16, 15, 1, padding=7),
                WNConv1d(16, 64, 41, 4, groups=4, padding=20),
                WNConv1d(64, 256, 41, 4, groups=16, padding=20),
                WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
                WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
                WNConv1d(1024, 1024, 5, 1, padding=2),
            ]
        )
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []
        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Module):
    """Complex multi-band spectrogram discriminator.

    Parameters
    ----------
    window_length : int
        Window length of the STFT.
    hop_factor : float, optional
        Hop length as a fraction of the window, by default 0.25.
    sample_rate : int, optional
        Sampling rate of the audio in Hz, by default 44100.
    bands : list, optional
        Frequency sub-bands to discriminate over, as fractions of Nyquist.
    """

    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44100,
        bands: list = BANDS,
    ):
        super().__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True,
        )

        n_fft = window_length // 2 + 1
        self.bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]

        ch = 32
        convs = lambda: nn.ModuleList(
            [
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = torch.view_as_real(x.stft())
        x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        return [x[..., b[0] : b[1]] for b in self.bands]

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        out = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            out.append(band)

        out = torch.cat(out, dim=-1)
        out = self.conv_post(out)
        fmap.append(out)
        return fmap


class Discriminator(ml.BaseModel):
    """One set of sub-discriminators per frequency branch.

    Parameters
    ----------
    rates : list, optional
        Resampling rates to run MSD at, by default ``[]`` (MSD disabled).
    periods : list, optional
        Periods to run MPD at, by default [2, 3, 5, 7, 11].
    fft_sizes : list, optional
        STFT window sizes to run MRD at, by default [2048, 1024, 512].
    sample_rates : List[int], optional
        Sampling rate of each branch.
    bands : list, optional
        Frequency sub-bands for MRD, by default ``BANDS``.
    """

    def __init__(
        self,
        rates: list = [],
        periods: list = [2, 3, 5, 7, 11],
        fft_sizes: list = [2048, 1024, 512],
        sample_rates: List[int] = [16000, 48000],
        bands: list = BANDS,
    ):
        super().__init__()
        self.sample_rates = sample_rates
        self.DISC = nn.ModuleList([])

        for sr in sample_rates:
            discs = []
            discs += [MPD(p) for p in periods]
            discs += [MSD(r, sample_rate=sr) for r in rates]
            discs += [MRD(f, sample_rate=sr, bands=bands) for f in fft_sizes]
            self.DISC.append(nn.ModuleList(discs))

    def preprocess(self, y):
        # Remove DC offset, then peak normalise.
        y = y - y.mean(dim=-1, keepdims=True)
        return 0.8 * y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

    def forward(self, x, input_sample_rate: int):
        x = self.preprocess(x)

        for i_sr in range(len(self.sample_rates)):
            if input_sample_rate == self.sample_rates[i_sr]:
                return [d(x) for d in self.DISC[i_sr]]

        raise ValueError(
            f"No discriminator for sample rate {input_sample_rate}; "
            f"expected one of {self.sample_rates}."
        )
