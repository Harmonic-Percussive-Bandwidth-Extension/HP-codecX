"""Convolution bookkeeping shared by the codec.

``CodecMixin`` reports the receptive-field delay and the output length of a
stack of convolutions, and can switch every convolution between padded and
unpadded operation. Adapted from Descript Audio Codec (MIT licence); the
single-stream ``compress`` / ``decompress`` helpers were dropped because they
do not apply to the branched architecture used here.
"""

import math

from torch import nn


class CodecMixin:
    @property
    def padding(self) -> bool:
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value: bool):
        assert isinstance(value, bool)

        layers = [
            l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in layers:
            if value:
                if hasattr(layer, "original_padding"):
                    layer.padding = layer.original_padding
            else:
                layer.original_padding = layer.padding
                layer.padding = tuple(0 for _ in range(len(layer.padding)))

        self._padding = value

    def get_delay(self) -> int:
        """Return the algorithmic delay, in samples.

        The delay is invariant to input length, so it is measured by running
        the length arithmetic backwards from a zero-length output.
        """
        l_out = self.get_output_length(0)
        L = l_out

        layers = [
            layer
            for layer in self.modules()
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        return (L - l_out) // 2

    def get_output_length(self, input_length: int) -> int:
        """Return the number of samples produced for ``input_length`` inputs."""
        L = input_length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L
