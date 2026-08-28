"""Reconstruct a test set with HP-codec (encode then decode, no prediction).

    python scripts/reconstruct.py \
        --path runs/hp-codec --model_tag best_finetuning \
        --input samples/input_16-48 --output samples/recons_16-48

Use this to measure the codec on its own. For bandwidth extension -- where the
48 kHz tokens are predicted rather than encoded -- use ``predict.py``.

The input directory is the one written by ``save_test_set.py``: files sort into
consecutive groups of ``len(sample_rates)``, one per branch.
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

from train_codec import DAC  # noqa: E402
from train_codec import Accelerator  # noqa: E402


def load_state(accel, tracker: Tracker, save_path: str, tag: str = "latest"):
    folder = f"{save_path}/{tag}"
    tracker.print(f"Loading codec from {Path('.').absolute()}/{folder}")
    if not (Path(folder) / "dac").exists():
        raise FileNotFoundError(f"No codec checkpoint at {folder}.")

    generator, _ = DAC.load_from_folder(folder=folder, map_location="cpu")
    return accel.prepare_model(generator)


@torch.no_grad()
def process(signal: List[AudioSignal], accel, generator):
    """Encode and decode one group of branch signals, matching input loudness."""
    signal = [sig.to(accel.device) for sig in signal]
    audio_data = [sig.audio_data for sig in signal]
    sample_rates = [sig.sample_rate for sig in signal]

    out = generator(audio_data, sample_rates, "full")

    return [
        AudioSignal(out["audio"][i], signal[i].sample_rate)
        .normalize(signal[i].loudness())
        .cpu()
        for i in range(len(sample_rates))
    ]


@argbind.bind(without_prefix=True)
@torch.no_grad()
def reconstruct(
    accel,
    path: str = "runs/hp-codec",
    input: str = "samples/input_16-48",
    output: str = "samples/recons_16-48",
    model_tag: str = "best_finetuning",
    sample_rates: List[int] = [16000, 48000],
):
    tracker = Tracker(log_file=f"{path}/eval.txt", rank=accel.local_rank)
    generator = load_state(accel, tracker, save_path=path, tag=model_tag)
    generator.eval()

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
            recons = process(signal, accel, generator)

            for j in range(len(recons)):
                recons[j].write(output / audio_files[i + j].name)

        tracker.done("test", f"N={len(audio_files)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            reconstruct(accel)
