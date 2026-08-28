"""Structure-informed residual vector quantisation.

``ResidualVectorQuantize`` holds one quantiser bank per frequency branch, and
each bank is split into two parallel sections -- ``harmonic`` and
``percussive``. Each section is an independent residual chain over the *same*
encoder output; their quantised outputs are summed before decoding. Training
alternates between ``full`` steps (both sections updated on unseparated audio)
and section steps (a single section updated on the corresponding component of
an HPSS decomposition), which is what drives the two sections apart.

``VectorQuantize`` is unchanged from Descript Audio Codec (MIT licence);
``ResidualVectorQuantize`` is the contribution of this work.
"""

from typing import List
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from hpcodec.nn.layers import WNConv1d

SECTIONS = ["harmonic", "percussive"]


class VectorQuantize(nn.Module):
    """Single vector quantiser with factorised, l2-normalised codes.

    Implementation follows Karpathy's deep-vector-quantization repository and
    the two Improved-VQGAN tricks (https://arxiv.org/pdf/2110.04627.pdf):

    1. Factorised codes: nearest-neighbour lookup happens in a low-dimensional
       space, which improves codebook usage.
    2. l2-normalised codes: turns the euclidean distance into a cosine
       similarity, which improves training stability.
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1, bias=False)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1, bias=False)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

        self.codebook.weight.data[0] = torch.zeros(codebook_dim)

    def forward(self, z: torch.Tensor):
        """Quantise ``z`` against the codebook.

        Parameters
        ----------
        z : torch.Tensor
            Continuous latent, shape ``[B, D, T]``.

        Returns
        -------
        z_q : torch.Tensor
            Quantised latent, shape ``[B, D, T]``, carrying straight-through
            gradients back to ``z``.
        commitment_loss : torch.Tensor
            Per-item commitment loss, shape ``[B]``.
        codebook_loss : torch.Tensor
            Per-item codebook loss, shape ``[B]``.
        indices : torch.Tensor
            Selected codebook entries, shape ``[B, T]``.
        z_e : torch.Tensor
            Projected latent before quantisation, shape ``[B, codebook_dim, T]``.
        """
        # Factorised codes (ViT-VQGAN): project into the low-dimensional space.
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # No-op forward, straight-through gradient estimator backward.
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # [N, D]

        # l2 normalise encodings and codebook (ViT-VQGAN).
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """Per-branch RVQ banks, each split into a harmonic and a percussive section.

    Parameters
    ----------
    input_dim : int
        Dimension of the encoder output fed to the quantisers.
    n_codebooks : List[int]
        Number of codebooks per frequency branch. Kept for reference and for
        checkpoint compatibility; the number actually instantiated per section
        is given by ``codebooks_comp``.
    codebooks_comp : List[int]
        Number of codebooks in the harmonic section and in the percussive
        section, in that order.
    codebook_size : int
        Number of entries per codebook.
    codebook_dim : Union[int, list]
        Dimension of the factorised code space. An ``int`` is broadcast over
        every branch and every codebook.
    quantizer_dropout : float
        Retained for configuration compatibility; unused by this quantiser.
    sample_rates : List[int]
        Sampling rate of each frequency branch, low to high.
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: List[int] = [3, 3],
        codebooks_comp: List[int] = [3, 3],
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        sample_rates: List[int] = [16000, 48000],
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [
                [codebook_dim for _ in range(max(codebooks_comp))]
                for _ in range(len(sample_rates))
            ]

        self.n_codebooks = n_codebooks
        self.codebooks_comp = codebooks_comp
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.sample_rates = sample_rates
        self.semantic_labels = list(SECTIONS)
        self.quantizer_dropout = quantizer_dropout

        self.RVQs = nn.ModuleList(
            nn.ModuleDict(
                {
                    label: nn.ModuleList(
                        VectorQuantize(input_dim, codebook_size, codebook_dim[i_sr][i])
                        for i in range(codebooks_comp[i_label])
                    )
                    for i_label, label in enumerate(self.semantic_labels)
                }
            )
            for i_sr in range(len(sample_rates))
        )

    def forward(
        self,
        z: List[torch.Tensor],
        input_sample_rates: List[int],
        semantic_phase: str = "full",
    ):
        """Quantise one encoder output per branch.

        Parameters
        ----------
        z : List[torch.Tensor]
            One ``[B, D, T]`` latent per entry of ``input_sample_rates``.
        input_sample_rates : List[int]
            Sampling rate of each entry of ``z``; used to select the matching
            quantiser bank.
        semantic_phase : str
            ``"full"`` runs both sections (each on its own residual chain, both
            contributing to the summed output). ``"harmonic"`` or
            ``"percussive"`` runs that section alone, which is how the
            structure-informed training steps update one section at a time.

        Returns
        -------
        z_q : List[torch.Tensor]
            Summed quantised latent per branch, shape ``[B, D, T]``.
        codes : List[dict]
            Per branch, ``{section: [B, n_codebooks, T]}`` codebook indices.
            Sections that did not run are absent.
        latents : List[dict]
            Per branch, ``{section: [B, n_codebooks, codebook_dim, T]}``
            pre-quantisation projections.
        commitment_losses, codebook_losses : List[torch.Tensor]
            Scalar losses per branch, summed over sections and codebooks.
        """
        z_q = []
        codes = []
        latents = []
        commitment_losses = []
        codebook_losses = []

        for i_sr in range(len(z)):
            for j_sr in range(len(self.sample_rates)):
                if input_sample_rates[i_sr] != self.sample_rates[j_sr]:
                    continue

                z_q_sr = 0
                residual_sr = z[i_sr]
                commitment_loss_sr = 0
                codebook_loss_sr = 0

                codebook_indices_sr = {label: [] for label in self.semantic_labels}
                latents_sr = {label: [] for label in self.semantic_labels}

                if semantic_phase == "full":
                    # Both sections quantise the same encoder output, each
                    # along its own residual chain; the two are summed.
                    for label in self.semantic_labels:
                        residual_i = residual_sr.clone()

                        for quantizer in self.RVQs[j_sr][label]:
                            z_q_i, commitment_i, codebook_i, indices_i, z_e_i = quantizer(
                                residual_i
                            )

                            z_q_sr = z_q_sr + z_q_i
                            residual_i = residual_i - z_q_i

                            commitment_loss_sr += commitment_i.mean()
                            codebook_loss_sr += codebook_i.mean()

                            codebook_indices_sr[label].append(indices_i)
                            latents_sr[label].append(z_e_i)
                else:
                    # Structure-informed step: a single section is exercised.
                    for quantizer in self.RVQs[j_sr][semantic_phase]:
                        z_q_i, commitment_i, codebook_i, indices_i, z_e_i = quantizer(
                            residual_sr
                        )

                        z_q_sr = z_q_sr + z_q_i
                        residual_sr = residual_sr - z_q_i

                        commitment_loss_sr += commitment_i.mean()
                        codebook_loss_sr += codebook_i.mean()

                        codebook_indices_sr[semantic_phase].append(indices_i)
                        latents_sr[semantic_phase].append(z_e_i)

                codes_sr = {
                    label: torch.stack(codebook_indices_sr[label], dim=1)
                    for label in self.semantic_labels
                    if codebook_indices_sr[label]
                }
                latents_sr = {
                    label: torch.stack(latents_sr[label], dim=1)
                    for label in self.semantic_labels
                    if latents_sr[label]
                }

                z_q.append(z_q_sr)
                codes.append(codes_sr)
                latents.append(latents_sr)
                commitment_losses.append(commitment_loss_sr)
                codebook_losses.append(codebook_loss_sr)

                break

        return z_q, codes, latents, commitment_losses, codebook_losses

    def from_codes(
        self,
        codes: List[dict],
        input_sample_rates: List[int],
    ) -> List[torch.Tensor]:
        """Rebuild the summed quantised latent of each branch from its codes.

        This is the inverse used at inference time, once HP-codecX has predicted
        the high-frequency tokens.

        Parameters
        ----------
        codes : List[dict]
            Per branch, ``{section: [B, n_codebooks, T]}`` codebook indices.
        input_sample_rates : List[int]
            Sampling rate of each entry of ``codes``.

        Returns
        -------
        List[torch.Tensor]
            One ``[B, D, T]`` latent per branch, ready for the decoder.
        """
        z_q = []

        for i_sr in range(len(codes)):
            for j_sr in range(len(self.sample_rates)):
                if input_sample_rates[i_sr] != self.sample_rates[j_sr]:
                    continue

                z_q_sr = 0.0
                sections = list(codes[i_sr].keys())
                n_codebooks = codes[i_sr][sections[0]].shape[1]

                for section in sections:
                    for i in range(n_codebooks):
                        quantizer = self.RVQs[j_sr][section][i]
                        z_p_i = quantizer.decode_code(codes[i_sr][section][:, i, :])
                        z_q_sr = z_q_sr + quantizer.out_proj(z_p_i)

                z_q.append(z_q_sr)
                break

        return z_q
