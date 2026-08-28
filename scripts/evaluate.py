"""Score reconstructed or extended audio against the reference test set.

    # bandwidth extension: every output is scored against the full-band reference
    python scripts/evaluate.py \
        --input samples/input_16-48 --output samples/hp-codecx --sr_comp 48000

    # codec reconstruction: each branch is scored against its own reference
    python scripts/evaluate.py \
        --input samples/input_16-48 --output samples/recons_16-48 \
        --reference per_branch

Writes ``metrics.csv`` inside the output directory and prints each metric with
a 95% pivotal bootstrap confidence interval.

The loss objects are instantiated with their default settings, which are the
ones the paper reports: two resolutions with window lengths {2048, 512}, and
mel-bin counts {150, 80}.
"""

import csv
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools import metrics
from audiotools.core import util
from audiotools.ml.decorators import Tracker

sys.path.append(os.getcwd())

from hpcodec.nn import loss as hp_losses  # noqa: E402

METRIC_COLUMNS = ["mel", "stft", "waveform", "sisdr", "visqol-audio"]


@dataclass
class State:
    stft_loss: hp_losses.MultiScaleSTFTLoss
    mel_loss: hp_losses.MelSpectrogramLoss
    waveform_loss: hp_losses.L1Loss
    sisdr_loss: hp_losses.SISDRLoss


def get_metrics(signal_path, recons_path, sr_comp: int, state: State) -> dict:
    """Compare one pair of files at a common sampling rate.

    Both signals are resampled to ``sr_comp`` and truncated to their shorter
    length, so that a decoder that emits a few extra samples is not penalised.
    """
    signal = AudioSignal(signal_path)
    recons = AudioSignal(recons_path)

    # Recorded before resampling, so rows can be grouped by branch.
    branch_sr = recons.sample_rate

    signal = signal.resample(sr_comp)
    recons = recons.resample(sr_comp)
    min_length = min(signal.signal_length, recons.signal_length)

    x = signal.clone().truncate_samples(min_length)
    y = recons.clone().truncate_samples(min_length)

    return {
        "mel": state.mel_loss(x, y),
        "stft": state.stft_loss(x, y),
        "waveform": state.waveform_loss(x, y),
        "sisdr": state.sisdr_loss(x, y),
        "visqol-audio": metrics.quality.visqol(x, y),
        "branch_sr": branch_sr,
        "recons_path": str(recons_path),
        "path": str(signal_path),
    }


def pivotal_bootstrap_ci(values, n_bootstrap: int = 10000, ci: float = 95):
    """Return the mean of ``values`` and its pivotal bootstrap interval.

    The pivotal (basic) interval reflects the bootstrap distribution back
    around the point estimate, so it is not forced to be symmetric.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    theta_hat = values.mean()
    rng = np.random.default_rng()
    boot_means = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(
        axis=1
    )

    alpha = (100 - ci) / 2
    q_low = np.percentile(boot_means, alpha)
    q_high = np.percentile(boot_means, 100 - alpha)

    return theta_hat, 2 * theta_hat - q_high, 2 * theta_hat - q_low


def summarize(rows: List[dict], sample_rates: List[int], ci: float = 95):
    """Print each metric, per branch, with its bootstrap confidence interval."""
    print()
    print(f"{'metric':<16}{'estimate':>12}{f'  {ci:.0f}% CI':>24}")
    print("-" * 52)

    for sr in sample_rates:
        rows_sr = [r for r in rows if r["branch_sr"] == sr]
        if not rows_sr:
            continue

        print(f"[{sr} Hz]  N={len(rows_sr)}")
        for column in METRIC_COLUMNS:
            theta, low, high = pivotal_bootstrap_ci([r[column] for r in rows_sr], ci=ci)
            print(f"{column:<16}{theta:>12.4f}{f'[{low:.4f}, {high:.4f}]':>24}")
        print()


@argbind.bind(without_prefix=True)
@torch.no_grad()
def evaluate(
    input: str = "samples/input_16-48",
    output: str = "samples/hp-codecx",
    n_proc: int = 8,
    sr_comp: int = 48000,
    reference: str = "full_band",
    sample_rates: List[int] = [16000, 48000],
):
    """Score every file in ``output`` against its reference in ``input``.

    Parameters
    ----------
    input : str
        Directory written by ``save_test_set.py``.
    output : str
        Directory of files to score; filenames must match those in ``input``.
    n_proc : int
        Worker processes. ViSQOL dominates the cost, so this scales well.
    sr_comp : int
        Common sampling rate the comparison is carried out at.
    reference : str
        ``"full_band"`` scores every branch against the highest-rate reference,
        which is what bandwidth extension should be judged on. ``"per_branch"``
        scores each branch against its own reference, for codec reconstruction.
    sample_rates : List[int]
        Branch sampling rates, in the order ``save_test_set.py`` wrote them.
    """
    if reference not in ("full_band", "per_branch"):
        raise ValueError(f"Unknown reference mode: {reference!r}")

    tracker = Tracker()
    state = State(
        waveform_loss=hp_losses.L1Loss(),
        stft_loss=hp_losses.MultiScaleSTFTLoss(),
        mel_loss=hp_losses.MelSpectrogramLoss(),
        sisdr_loss=hp_losses.SISDRLoss(),
    )

    audio_files = util.find_audio(input)
    audio_files.sort()

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    rows = []

    @tracker.track("metrics", len(audio_files))
    def record(future, writer):
        row = future.result()
        for k, v in row.items():
            if torch.is_tensor(v):
                row[k] = v.item()
        writer.writerow(row)
        rows.append(dict(row))
        row.pop("path")
        return row

    n_branches = len(sample_rates)
    futures = []

    with tracker.live:
        with open(output / "metrics.csv", "w") as csvfile:
            with ProcessPoolExecutor(n_proc, mp.get_context("fork")) as pool:
                # Files come in groups of n_branches: one per branch, same excerpt.
                for i in range(0, len(audio_files), n_branches):
                    for j in range(n_branches):
                        ref = (
                            audio_files[i + n_branches - 1]
                            if reference == "full_band"
                            else audio_files[i + j]
                        )
                        futures.append(
                            pool.submit(
                                get_metrics,
                                ref,
                                output / audio_files[i + j].name,
                                sr_comp,
                                state,
                            )
                        )

                writer = csv.DictWriter(
                    csvfile, fieldnames=list(futures[0].result().keys())
                )
                writer.writeheader()

                for future in futures:
                    record(future, writer)

        tracker.done("test", f"N={len(audio_files)}")

    summarize(rows, sample_rates)


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        evaluate()
