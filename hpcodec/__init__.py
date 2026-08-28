"""Harmonic-Percussive disentangled neural audio codec for bandwidth extension.

This package provides the two models described in the paper:

``hpcodec.model.DAC``
    HP-codec, a branched neural audio codec whose latent space is disentangled
    both across frequency bands (a 16 kHz branch and a 48 kHz branch) and across
    harmonic / percussive structure (two parallel RVQ sections per branch).

``hpcodec.model.TransformerModel``
    HP-codecX, the audio language model that predicts the 48 kHz branch tokens
    from the 16 kHz branch tokens, thereby performing bandwidth extension.

The architecture and the training loop are derived from Descript Audio Codec
(https://github.com/descriptinc/descript-audio-codec, MIT licence). See the
NOTICE section of the README for details.
"""

__version__ = "1.0.0"

import audiotools

# Model checkpoints are serialised with ``torch.package``. Interning this
# package embeds its source inside the checkpoint, which makes the released
# weights loadable without this repository being importable.
audiotools.ml.BaseModel.INTERN += ["hpcodec.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]

from . import model
from . import nn
from .model import DAC
from .model import Discriminator
from .model import TransformerModel

__all__ = ["DAC", "Discriminator", "TransformerModel", "model", "nn"]
