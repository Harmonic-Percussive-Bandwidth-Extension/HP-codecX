"""Bandwidth extension with HP-codecX: 16 kHz in, 48 kHz out.

    python scripts/predict.py \
        --path runs/hp-codecx --model_tag best \
        --codec_path runs/hp-codec/best_finetuning \
        --input samples/input_16-48 --output samples/hp-codecx

Only the 16 kHz file of each pair is used as input. Its harmonic and percussive
tokens are extracted with the frozen codec, the transformers predict the three
48 kHz codebooks of each section, and the 48 kHz decoder synthesises the
high-frequency content, which is added back to the upsampled input.

The 48 kHz file of each pair is never read here; it is the reference that
``evaluate.py`` scores against.
"""

import os
import sys
from pathlib import Path
from typing import List

import argbind
import torch
from audiotools import AudioSignal
from audiotools.core import util
from audiotools.ml.decorators import Tracker

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from train_codecx import DAC  # noqa: E402
from train_codecx import Accelerator  # noqa: E402
from train_codecx import TransformerModel  # noqa: E402
from train_codecx import decode_prediction  # noqa: E402
from train_codecx import sample_tokens  # noqa: E402


def load_state(
    accel,
    tracker: Tracker,
    save_path: str,
    codec_save_path: str,
    tag: str = "best",
):
    tracker.print(f"Loading codec from {Path('.').absolute()}/{codec_save_path}")
    if not (Path(codec_save_path) / "dac").exists():
        raise FileNotFoundError(f"No codec checkpoint at {codec_save_path}.")
    generator, _ = DAC.load_from_folder(folder=codec_save_path, map_location="cpu")

    folder = f"{save_path}/{tag}"
    tracker.print(f"Loading language model from {Path('.').absolute()}/{folder}")
    if not (Path(folder) / "transformermodel").exists():
        raise FileNotFoundError(f"No HP-codecX checkpoint at {folder}.")
    transformer, _ = TransformerModel.load_from_folder(
        folder=folder, map_location="cpu"
    )

    return accel.prepare_model(generator), accel.prepare_model(transformer)


@torch.no_grad()
def process(signal: List[AudioSignal], accel, generator, transformer, top_p: float):
    """Extend one group of branch signals from its lowest branch."""
    signal = [sig.to(accel.device) for sig in signal]
    audio_data = [sig.audio_data for sig in signal]
    sample_rates = [sig.sample_rate for sig in signal]
    lengths = [data.size()[-1] for data in audio_data]

    # The codec is run on every branch, but only the low-branch tokens are fed
    # to the transformers; the high-branch tokens are the ones being predicted.
    codes = generator(audio_data, sample_rates, "full")["codes"]

    predicted = sample_tokens(
        transformer,
        codes,
        n_low_branches=len(sample_rates) - 1,
        n_pred_tokens=transformer.n_pred_tokens,
        top_p=top_p,
    )
    recons = decode_prediction(generator, predicted, signal[0], sample_rates, lengths)

    return [
        AudioSignal(recons[i], signal[i].sample_rate).cpu()
        for i in range(len(sample_rates))
    ]


@argbind.bind(without_prefix=True)
@torch.no_grad()
def predict(
    accel,
    path: str = "runs/hp-codecx",
    codec_path: str = "runs/hp-codec/best_finetuning",
    input: str = "samples/input_16-48",
    output: str = "samples/hp-codecx",
    model_tag: str = "best",
    top_p: float = 0.95,
    sample_rates: List[int] = [16000, 48000],
):
    tracker = Tracker(log_file=f"{path}/eval.txt", rank=accel.local_rank)
    generator, transformer = load_state(
        accel, tracker, save_path=path, codec_save_path=codec_path, tag=model_tag
    )
    generator.eval()
    transformer.eval()

    audio_files = util.find_audio(input)
    audio_files.sort()

    global process
    process = tracker.track("process", len(audio_files))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with tracker.live:
        # Files come in groups of len(sample_rates): one per branch, same excerpt.
        for i in range(0, len(audio_files), len(sample_rates)):
            signal = [AudioSignal(audio_files[i + j]) for j in range(len(sample_rates))]
            recons = process(signal, accel, generator, transformer, top_p)

            for j in range(len(recons)):
                recons[j].write(output / audio_files[i + j].name)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            predict(accel)
