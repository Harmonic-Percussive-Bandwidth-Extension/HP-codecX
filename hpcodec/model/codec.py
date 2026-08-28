"""HP-codec: the frequency- and structure-disentangled neural audio codec.

The DAC architecture is replicated into one branch per sampling rate. The
lowest branch encodes and reconstructs the signal directly; every higher branch
encodes only the *residual* between the band-limited target at its own rate and
the upsampled reconstruction of the branch below it. That residual coupling is
what makes the high-frequency tokens predictable from the low-frequency ones,
which is the property HP-codecX exploits.

Within a branch, quantisation is further split into a harmonic and a percussive
section (see ``hpcodec.nn.quantize``).

Derived from Descript Audio Codec (MIT licence); the branching, the residual
coupling between branches and the harmonic/percussive split are the
contributions of this work.
"""

import math
from typing import List
from typing import Union

import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.ml import BaseModel
from torch import nn

from hpcodec.nn.layers import Snake1d
from hpcodec.nn.layers import WNConv1d
from hpcodec.nn.layers import WNConvTranspose1d
from hpcodec.nn.quantize import ResidualVectorQuantize

from .base import CodecMixin


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """One convolutional encoder per frequency branch.

    Parameters
    ----------
    d_models : List[int]
        Base channel count of each branch, doubled by every encoder block.
    strides : List[list]
        Downsampling factors of each branch, giving its hop length.
    d_latents : List[int]
        Output dimension of each branch.
    sample_rates : List[int]
        Sampling rate each branch operates at, low to high.
    """

    def __init__(
        self,
        d_models: List[int] = [64, 64],
        strides: List[list] = [[2, 2, 5, 8], [2, 5, 6, 8]],
        d_latents: List[int] = [1024, 1024],
        sample_rates: List[int] = [16000, 48000],
    ):
        super().__init__()
        self.branches = nn.ModuleList([])
        self.enc_dim = []
        self.sample_rates = sample_rates

        # Copied, not mutated in place: `BaseModel` records the constructor
        # kwargs after construction, so mutating the caller's list would write
        # the already-doubled channel counts into the checkpoint metadata and
        # make the model impossible to rebuild from it.
        d_models = list(d_models)

        for i_sr in range(len(sample_rates)):
            block = [WNConv1d(1, d_models[i_sr], kernel_size=7, padding=3)]

            # Encoder blocks double the channel count as they downsample.
            for stride in strides[i_sr]:
                d_models[i_sr] *= 2
                block += [EncoderBlock(d_models[i_sr], stride=stride)]

            block += [
                Snake1d(d_models[i_sr]),
                WNConv1d(d_models[i_sr], d_latents[i_sr], kernel_size=3, padding=1),
            ]

            self.branches.append(nn.Sequential(*block))
            self.enc_dim += [d_models[i_sr]]

    def forward(self, x: List[torch.Tensor], input_sample_rates: Union[int, List[int]]):
        if isinstance(input_sample_rates, int):
            input_sample_rates = [input_sample_rates] * len(x)

        y = []
        for i_sr in range(len(x)):
            for j_sr in range(len(self.sample_rates)):
                if input_sample_rates[i_sr] == self.sample_rates[j_sr]:
                    y.append(self.branches[j_sr](x[i_sr]))
                    break
        return y


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        # An odd stride needs an odd kernel to keep the transposed convolution
        # length aligned with the encoder's.
        kernel = 2 * stride + 1 if stride % 2 == 1 else 2 * stride
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=kernel,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    """One convolutional decoder per frequency branch."""

    def __init__(
        self,
        input_channels: List[int],
        channels: List[int],
        rates: List[list],
        sample_rates: List[int],
        d_out: int = 1,
    ):
        super().__init__()
        self.decoders = nn.ModuleList([])
        self.sample_rates = sample_rates

        for i_sr in range(len(sample_rates)):
            layers = [
                WNConv1d(input_channels[i_sr], channels[i_sr], kernel_size=7, padding=3)
            ]

            for i, stride in enumerate(rates[i_sr]):
                input_dim = channels[i_sr] // 2**i
                output_dim = channels[i_sr] // 2 ** (i + 1)
                layers += [DecoderBlock(input_dim, output_dim, stride)]

            layers += [
                Snake1d(output_dim),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
                nn.Tanh(),
            ]

            self.decoders.append(nn.Sequential(*layers))

    def forward(self, x: List[torch.Tensor], input_sample_rates: Union[int, List[int]]):
        if isinstance(input_sample_rates, int):
            input_sample_rates = [input_sample_rates] * len(x)

        y = []
        for i_sr in range(len(x)):
            if x[i_sr] is None:
                y.append(None)
                continue
            for j_sr in range(len(self.sample_rates)):
                if input_sample_rates[i_sr] == self.sample_rates[j_sr]:
                    y.append(self.decoders[j_sr](x[i_sr]))
                    break
        return y


class DAC(BaseModel, CodecMixin):
    """HP-codec.

    The class name is kept as ``DAC`` so that released checkpoints -- which are
    serialised with ``torch.package`` under the class name -- keep loading, and
    so that configuration keys stay ``DAC.*``.

    Parameters
    ----------
    encoder_dims : List[int]
        Base channel count of each encoder branch.
    encoder_rates : List[List[int]]
        Downsampling factors per branch. Their product is the branch hop
        length; picking them so that the branches share a bitrate is what makes
        the token sequences time-aligned across branches.
    latent_dims : List[int], optional
        Encoder output dimension per branch. Defaults to
        ``encoder_dims[i] * 2 ** len(encoder_rates[i])``.
    decoder_dims : List[int]
        Base channel count of each decoder branch.
    decoder_rates : List[List[int]]
        Upsampling factors per branch, mirroring ``encoder_rates``.
    n_codebooks : List[int]
        Number of codebooks per branch.
    codebooks_comp : List[int], optional
        Number of codebooks in the harmonic and percussive sections. Defaults
        to ``n_codebooks[0]`` for both.
    codebook_size : int
        Number of entries per codebook.
    codebook_dim : Union[int, list]
        Dimension of the factorised code space.
    quantizer_dropout : bool
        Retained for configuration compatibility; unused.
    sample_rates : List[int]
        Sampling rate of each branch, low to high.
    final_sum : bool
        If True, each higher branch adds the upsampled reconstruction of the
        branch below to its own output, so that branch outputs are full-band
        signals rather than residuals.
    """

    def __init__(
        self,
        encoder_dims: List[int] = [64, 64],
        encoder_rates: List[List[int]] = [[2, 2, 5, 8], [2, 5, 6, 8]],
        latent_dims: List[int] = None,
        decoder_dims: List[int] = [1536, 1536],
        decoder_rates: List[List[int]] = [[8, 5, 2, 2], [8, 6, 5, 2]],
        n_codebooks: List[int] = [3, 3],
        codebooks_comp: List[int] = None,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: bool = False,
        sample_rates: List[int] = [16000, 48000],
        final_sum: bool = False,
    ):
        super().__init__()

        self.encoder_dims = encoder_dims
        self.encoder_rates = encoder_rates
        self.decoder_dims = decoder_dims
        self.decoder_rates = decoder_rates
        self.sample_rates = sample_rates
        self.final_sum = final_sum

        if latent_dims is None:
            latent_dims = [
                encoder_dims[i_sr] * (2 ** len(encoder_rates[i_sr]))
                for i_sr in range(len(sample_rates))
            ]
        self.latent_dims = latent_dims

        self.hop_lengths = [
            np.prod(encoder_rates[i_sr]) for i_sr in range(len(sample_rates))
        ]

        self.n_codebooks = n_codebooks
        if codebooks_comp is None:
            self.codebooks_comp = [n_codebooks[0], n_codebooks[0]]
        else:
            self.codebooks_comp = codebooks_comp
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.encoder = Encoder(encoder_dims, encoder_rates, latent_dims, sample_rates)
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dims[0],
            n_codebooks=n_codebooks,
            codebooks_comp=self.codebooks_comp,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            sample_rates=sample_rates,
        )
        self.decoder = Decoder(
            latent_dims,
            decoder_dims,
            decoder_rates,
            sample_rates,
        )

        self.apply(init_weights)
        self.delay = self.get_delay()

    def preprocess(
        self, audio_data: List[torch.Tensor], sample_rates: List[int]
    ) -> List[torch.Tensor]:
        """Right-pad each branch input to a whole number of hop lengths."""
        if sample_rates is None:
            sample_rates = self.sample_rates

        right_pads = [
            math.ceil(audio_data[i_sr].shape[-1] / self.hop_lengths[i_sr])
            * self.hop_lengths[i_sr]
            - audio_data[i_sr].shape[-1]
            for i_sr in range(len(sample_rates))
        ]
        return [
            nn.functional.pad(audio_data[i_sr], (0, right_pads[i_sr]))
            for i_sr in range(len(sample_rates))
        ]

    def encode(
        self,
        audio_data: List[torch.Tensor],
        input_sample_rates: List[int],
        semantic_phase: str = "full",
    ):
        """Encode and quantise one waveform per branch.

        Parameters
        ----------
        audio_data : List[torch.Tensor]
            One ``[B, 1, T]`` waveform per entry of ``input_sample_rates``.
        input_sample_rates : List[int]
            Sampling rate of each entry of ``audio_data``.
        semantic_phase : str
            ``"full"``, ``"harmonic"`` or ``"percussive"``; selects which RVQ
            sections run. See ``hpcodec.nn.quantize.ResidualVectorQuantize``.

        Returns
        -------
        z_q, codes, latents, commitment_losses, codebook_losses
            As returned by the quantiser, one entry per branch.
        """
        z = self.encoder(audio_data, input_sample_rates)
        return self.quantizer(z, input_sample_rates, semantic_phase)

    def decode(self, z: List[torch.Tensor], input_sample_rates: List[int]):
        """Decode one quantised latent per branch back to a waveform."""
        return self.decoder(z, input_sample_rates)

    def forward(
        self,
        audio_data: Union[torch.Tensor, List[torch.Tensor]],
        input_sample_rates: Union[int, List[int]] = None,
        semantic_phase: str = "full",
    ) -> dict:
        """Run the full cascade over every branch.

        Branch 0 encodes its input directly. Every subsequent branch encodes
        the difference between its own input and the upsampled reconstruction
        of the branch below, so it only has to represent the frequency content
        the lower branch could not carry.

        Parameters
        ----------
        audio_data : Union[torch.Tensor, List[torch.Tensor]]
            One ``[B, 1, T]`` waveform per branch, band-limited and sampled at
            the corresponding rate.
        input_sample_rates : Union[int, List[int]]
            Sampling rate of each entry of ``audio_data``.
        semantic_phase : str
            ``"full"``, ``"harmonic"`` or ``"percussive"``.

        Returns
        -------
        dict
            ``audio`` (reconstruction per branch, trimmed to the input length),
            ``z`` (quantised latents), ``codes`` (per-section token indices),
            ``latents`` (pre-quantisation projections), and the per-branch
            ``vq/commitment_losses`` and ``vq/codebook_losses``.
        """
        if isinstance(audio_data, torch.Tensor):
            audio_data = [audio_data]
        if isinstance(input_sample_rates, int):
            input_sample_rates = [input_sample_rates]

        lengths = [
            audio_data[i_sr].shape[-1] for i_sr in range(len(input_sample_rates))
        ]
        audio_data = self.preprocess(audio_data, input_sample_rates)

        z_q = []
        codes = []
        latents = []
        commitment_losses = []
        codebook_losses = []
        x = []

        for i_sr in range(len(audio_data)):
            if i_sr == 0:
                branch_input = audio_data[i_sr]
                lower_freq_sig = None
            else:
                # Upsample the reconstruction of the branch below and subtract
                # it, so this branch only encodes what is missing from it.
                lower_freq_sig = (
                    AudioSignal(
                        x[i_sr - 1][..., : lengths[i_sr - 1]],
                        input_sample_rates[i_sr - 1],
                    )
                    .clone()
                    .resample(input_sample_rates[i_sr])
                    .audio_data
                )

                right_pad = audio_data[i_sr].shape[-1] - lower_freq_sig.shape[-1]
                if right_pad > 0:
                    lower_freq_sig = nn.functional.pad(lower_freq_sig, (0, right_pad))
                elif right_pad < 0:
                    lower_freq_sig = lower_freq_sig[:, :, :right_pad]

                branch_input = audio_data[i_sr] - lower_freq_sig

            (
                z_q_sr,
                codes_sr,
                latents_sr,
                commitment_losses_sr,
                codebook_losses_sr,
            ) = self.encode([branch_input], [input_sample_rates[i_sr]], semantic_phase)

            x_sr = self.decode(z_q_sr, [input_sample_rates[i_sr]])

            if i_sr > 0 and self.final_sum:
                x_sr[0] = x_sr[0] + lower_freq_sig

            z_q.append(z_q_sr[0])
            codes.append(codes_sr[0])
            latents.append(latents_sr[0])
            commitment_losses.append(commitment_losses_sr[0])
            codebook_losses.append(codebook_losses_sr[0])
            x.append(x_sr[0])

        return {
            "audio": [x[i_sr][..., : lengths[i_sr]] for i_sr in range(len(x))],
            "z": z_q,
            "codes": codes,
            "latents": latents,
            "vq/commitment_losses": commitment_losses,
            "vq/codebook_losses": codebook_losses,
        }


if __name__ == "__main__":
    from functools import partial

    model = DAC().to("cpu")

    for _, m in model.named_modules():
        o = m.extra_repr()
        p = sum(np.prod(p.size()) for p in m.parameters())
        setattr(m, "extra_repr", partial(lambda o, p: o + f" {p / 1e6:<.3f}M params.", o=o, p=p))

    print(model)
    print("Total # of params: ", sum(np.prod(p.size()) for p in model.parameters()))
