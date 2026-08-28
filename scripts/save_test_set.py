"""Export a paired test set to disk, one file per branch sampling rate.

    python scripts/save_test_set.py --args.load conf/codec/16khz.yml \
        --output samples/input_16-48

Each test item is written once per branch as ``sample_<i>_sr<rate>.wav``, so
``sample_3_sr16000.wav`` is the 8 kHz-limited input and ``sample_3_sr48000.wav``
is its full-band reference. ``reconstruct.py``, ``predict.py`` and
``evaluate.py`` all rely on that pairing and on the files sorting together.
"""

import csv
import os
import sys
from pathlib import Path
from typing import List

import argbind
import torch
from audiotools.ml.decorators import Tracker

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

import train_codec  # noqa: E402
from train_codec import Accelerator  # noqa: E402


@torch.no_grad()
def process(item, test_data, sample_rates: List[int]):
    """Apply the test transform to one item, at every branch sampling rate."""
    signal = {}
    for sr in sample_rates:
        signal[sr] = test_data.transform(
            item[str(sr)].clone(), **item["transform_args"]
        )
        signal[sr].cpu()
    return signal


@argbind.bind(without_prefix=True)
@torch.no_grad()
def save_test_set(
    args,
    accel,
    sample_rates: List[int] = [16000, 48000],
    output: str = "samples/input_16-48",
):
    tracker = Tracker()
    with argbind.scope(args, "test"):
        test_data = train_codec.build_dataset(sample_rates)

    global process
    process = tracker.track("process", len(test_data))(process)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    with open(output / "metadata.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["path", "original"])
        writer.writeheader()

        with tracker.live:
            for i in range(len(test_data)):
                signal = process(test_data[i], test_data, sample_rates)
                for sr in sample_rates:
                    path = output / f"sample_{i}_sr{sr}.wav"
                    writer.writerow(
                        {
                            "path": str(path),
                            "original": str(signal[sr].path_to_input_file),
                        }
                    )
                    signal[sr].write(path)

            tracker.done("test", f"N={len(test_data)}")


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        with Accelerator() as accel:
            save_test_set(args, accel)
