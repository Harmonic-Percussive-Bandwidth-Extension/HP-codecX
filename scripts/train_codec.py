"""Train HP-codec.

Training runs in three phases, one invocation each, all writing to the same
``--save_path``:

    python scripts/train_codec.py --args.load conf/codec/16khz.yml    --save_path runs/hp-codec
    python scripts/train_codec.py --args.load conf/codec/48khz.yml    --save_path runs/hp-codec
    python scripts/train_codec.py --args.load conf/codec/finetune.yml --save_path runs/hp-codec

The phase is selected by ``training_sr``: a sampling rate trains that branch
(resuming from, and freezing, every branch below it), and ``"finetuning"``
unfreezes everything and trains the branches jointly.

Within a phase, each step draws one of three objectives with equal probability:
``full`` updates both RVQ sections on unseparated audio, while ``harmonic`` and
``percussive`` update a single section on the corresponding component of an
HPSS decomposition. That alternation is what drives the two sections apart.

Adapted from the Descript Audio Codec training script (MIT licence).
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Union

import argbind
import librosa
import torch
from audiotools import AudioSignal
from audiotools import ml
from audiotools.core import util
from audiotools.data import transforms
from audiotools.data.datasets import AudioDataset
from audiotools.data.datasets import AudioLoader
from audiotools.ml.decorators import Tracker
from audiotools.ml.decorators import timer
from audiotools.ml.decorators import when
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

import hpcodec  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# Enable the cudnn autotuner (can be altered by the funcs.seed function).
torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))

# The three objectives one training step can be drawn from.
SEMANTIC_LABELS = ["full", "harmonic", "percussive"]

# HPSS parameters. A high margin and power give a strongly discriminative
# separation, so the harmonic component is almost purely harmonic and vice
# versa. The two components then do not sum back to the input; the discarded
# residual is preserved by also training on the unseparated signal.
HPSS_MARGIN = 3.0
HPSS_POWER = 2.0

# Optimizers
AdamW = argbind.bind(torch.optim.AdamW, "generator", "discriminator")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("generator", "discriminator")
def ExponentialLR(optimizer, gamma: float = 1.0):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)


# Models
DAC = argbind.bind(hpcodec.model.DAC)
Discriminator = argbind.bind(hpcodec.model.Discriminator)

# Data
AudioDataset = argbind.bind(AudioDataset, "train", "val")
AudioLoader = argbind.bind(AudioLoader, "train", "val")

# Transforms
filter_fn = lambda fn: hasattr(fn, "transform") and fn.__qualname__ not in [
    "BaseTransform",
    "Compose",
    "Choose",
]
tfm = argbind.bind_module(transforms, "train", "val", filter_fn=filter_fn)

# Losses
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(hpcodec.nn.loss, filter_fn=filter_fn)


class MultiDataset(AudioDataset):
    """Serves the same excerpt at every branch sampling rate, plus its HPSS parts.

    ``datasets`` maps a sampling rate to an ``AudioDataset`` reading the same
    sources, so index ``idx`` refers to the same audio everywhere. Each item is
    a dict keyed by ``str(sample_rate)``, with ``"<sr>_harmonic"`` and
    ``"<sr>_percussive"`` alongside. The decomposition is computed once on the
    highest rate and resampled down, so the components stay consistent across
    branches.
    """

    def __init__(self, datasets: dict):
        self.datasets = datasets

    def __len__(self):
        keys = list(self.datasets.keys())
        return len(self.datasets[keys[0]])

    def __getitem__(self, idx):
        keys = list(self.datasets.keys())

        items = self.datasets[keys[0]][idx]
        items[str(keys[0])] = items.pop("signal")
        for key in keys[1:]:
            items[str(key)] = self.datasets[key][idx]["signal"]

        highest = str(keys[-1])
        harmonic, percussive = librosa.effects.hpss(
            items[highest].audio_data.numpy(), margin=HPSS_MARGIN, power=HPSS_POWER
        )
        items[f"{highest}_harmonic"] = AudioSignal(
            torch.Tensor(harmonic), items[highest].sample_rate
        )
        items[f"{highest}_percussive"] = AudioSignal(
            torch.Tensor(percussive), items[highest].sample_rate
        )

        for key in keys[:-1]:
            for section in ("harmonic", "percussive"):
                items[f"{key}_{section}"] = (
                    items[f"{highest}_{section}"]
                    .clone()
                    .resample(items[str(key)].sample_rate)
                )

        return items


def get_infinite_loader(dataloader):
    while True:
        for batch in dataloader:
            yield batch


@argbind.bind("train", "val")
def build_transform(
    augment_prob: float = 1.0,
    preprocess: list = ["Identity"],
    augment: list = ["Identity"],
    postprocess: list = ["Identity"],
):
    to_tfm = lambda l: [getattr(tfm, x)() for x in l]
    preprocess = transforms.Compose(*to_tfm(preprocess), name="preprocess")
    augment = transforms.Compose(*to_tfm(augment), name="augment", prob=augment_prob)
    postprocess = transforms.Compose(*to_tfm(postprocess), name="postprocess")
    return transforms.Compose(preprocess, augment, postprocess)


@argbind.bind("train", "val", "test")
def build_dataset(
    sample_rates: List[int] = [16000, 48000],
    folders: dict = None,
):
    """Build one dataset per branch sampling rate over the same source folders."""
    datasets = {}
    for _, sources in folders.items():
        for sr in sample_rates:
            loader = AudioLoader(sources=sources)
            transform = build_transform()
            datasets[sr] = AudioDataset(loader, sr, transform=transform)

    dataset = MultiDataset(datasets)
    dataset.transform = transform
    return dataset


@dataclass
class State:
    generator: DAC
    optimizer_g: AdamW
    scheduler_g: ExponentialLR

    discriminator: Discriminator
    optimizer_d: AdamW
    scheduler_d: ExponentialLR

    stft_loss: losses.MultiScaleSTFTLoss
    mel_loss: losses.MelSpectrogramLoss
    gan_loss: losses.GANLoss
    waveform_loss: losses.L1Loss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


def working_sample_rates(sample_rates: List[int], training_sr: Union[int, str]):
    """Return the branches active in this phase, and the index of the trained one.

    During finetuning every branch is active and none is singled out, so the
    index is ``None``.
    """
    if training_sr == "finetuning":
        return sample_rates, None

    working = [sr for sr in sample_rates if sr <= training_sr]
    return working, len(working) - 1


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    save_path: str,
    resume: bool = False,
    tag: str = "latest",
    load_weights: bool = False,
):
    generator, discriminator = None, None

    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        tracker.print(f"Resuming from {str(Path('.').absolute())}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "dac").exists():
            generator, _ = DAC.load_from_folder(**kwargs)
        if (Path(kwargs["folder"]) / "discriminator").exists():
            discriminator, _ = Discriminator.load_from_folder(**kwargs)

    generator = DAC() if generator is None else generator
    discriminator = Discriminator() if discriminator is None else discriminator

    tracker.print(generator)
    tracker.print(discriminator)

    generator = accel.prepare_model(generator)
    discriminator = accel.prepare_model(discriminator)

    with argbind.scope(args, "generator"):
        optimizer_g = AdamW(generator.parameters(), use_zero=accel.use_ddp)
        scheduler_g = ExponentialLR(optimizer_g)
    with argbind.scope(args, "discriminator"):
        optimizer_d = AdamW(discriminator.parameters(), use_zero=accel.use_ddp)
        scheduler_d = ExponentialLR(optimizer_d)

    sample_rates = accel.unwrap(generator).sample_rates
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rates)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rates)

    return State(
        generator=generator,
        optimizer_g=optimizer_g,
        scheduler_g=scheduler_g,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        waveform_loss=losses.L1Loss(),
        stft_loss=losses.MultiScaleSTFTLoss(),
        mel_loss=losses.MelSpectrogramLoss(),
        gan_loss=losses.GANLoss(discriminator),
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


@timer()
@torch.no_grad()
def val_loop(batch, state, accel, training_sr):
    state.generator.eval()
    batch = util.prepare_batch(batch, accel.device)

    sample_rates = accel.unwrap(state.generator).sample_rates
    working_sr, i_training_sr = working_sample_rates(sample_rates, training_sr)

    signal = [
        state.val_data.transform(batch[str(sr)].clone(), **batch["transform_args"])
        for sr in working_sr
    ]
    audio_data = [sig.audio_data for sig in signal]

    out = state.generator(audio_data, working_sr, "full")
    recons = [
        AudioSignal(out["audio"][i], signal[i].sample_rate)
        for i in range(len(working_sr))
    ]

    output = {}
    if i_training_sr is not None:
        output[f"stft/loss_{training_sr}"] = state.stft_loss(
            recons[i_training_sr], signal[i_training_sr]
        )
        output[f"mel/loss_{training_sr}"] = state.mel_loss(
            recons[i_training_sr], signal[i_training_sr]
        )
        output[f"waveform/loss_{training_sr}"] = state.waveform_loss(
            recons[i_training_sr], signal[i_training_sr]
        )
    else:
        # Finetuning: every branch contributes, and the per-branch terms are
        # summed into the aggregate the tracker follows.
        for name, loss_fn in (
            ("stft", state.stft_loss),
            ("mel", state.mel_loss),
            ("waveform", state.waveform_loss),
        ):
            per_branch = [
                loss_fn(recons[i], signal[i]) for i in range(len(working_sr))
            ]
            for i, sr in enumerate(working_sr):
                output[f"{name}/loss_{training_sr}_{sr}"] = per_branch[i]
            output[f"{name}/loss_{training_sr}"] = sum(per_branch)

    output[f"loss_{training_sr}"] = output[f"mel/loss_{training_sr}"]
    return output


@timer()
def train_loop(state, batch, accel, lambdas, training_sr, training_label):
    state.generator.train()
    state.discriminator.train()
    output = {}

    batch = util.prepare_batch(batch, accel.device)

    with torch.no_grad():
        sample_rates = accel.unwrap(state.generator).sample_rates
        working_sr, i_training_sr = working_sample_rates(sample_rates, training_sr)

        # `full` steps see the unseparated signal, section steps see the
        # matching HPSS component.
        suffix = "" if training_label == "full" else f"_{training_label}"
        signal = [
            state.train_data.transform(
                batch[f"{sr}{suffix}"].clone(), **batch["transform_args"]
            )
            for sr in working_sr
        ]
        audio_data = [sig.audio_data for sig in signal]

    with accel.autocast():
        out = state.generator(audio_data, working_sr, training_label)
        recons = [
            AudioSignal(out["audio"][i], signal[i].sample_rate)
            for i in range(len(working_sr))
        ]
        commitment_losses = out["vq/commitment_losses"]
        codebook_losses = out["vq/codebook_losses"]

    # --- Discriminator step -------------------------------------------------
    with accel.autocast():
        if i_training_sr is not None:
            output[f"adv/disc_loss_{training_sr}"] = state.gan_loss.discriminator_loss(
                recons[i_training_sr], signal[i_training_sr], training_sr
            )
        else:
            per_branch = [
                state.gan_loss.discriminator_loss(recons[i], signal[i], sr)
                for i, sr in enumerate(working_sr)
            ]
            for i, sr in enumerate(working_sr):
                output[f"adv/disc_loss_{training_sr}_{sr}"] = per_branch[i]
            output[f"adv/disc_loss_{training_sr}"] = sum(per_branch)

    state.optimizer_d.zero_grad()
    accel.backward(output[f"adv/disc_loss_{training_sr}"])
    accel.scaler.unscale_(state.optimizer_d)
    output[f"other/grad_norm_d_{training_sr}"] = torch.nn.utils.clip_grad_norm_(
        state.discriminator.parameters(), 10.0
    )
    accel.step(state.optimizer_d)
    state.scheduler_d.step()

    # --- Generator step -----------------------------------------------------
    with accel.autocast():
        if i_training_sr is not None:
            i = i_training_sr
            output[f"stft/loss_{training_sr}"] = state.stft_loss(recons[i], signal[i])
            output[f"mel/loss_{training_sr}"] = state.mel_loss(recons[i], signal[i])
            output[f"waveform/loss_{training_sr}"] = state.waveform_loss(
                recons[i], signal[i]
            )
            (
                output[f"adv/gen_loss_{training_sr}"],
                output[f"adv/feat_loss_{training_sr}"],
            ) = state.gan_loss.generator_loss(recons[i], signal[i], training_sr)
            output[f"vq/commitment_loss_{training_sr}"] = commitment_losses[i]
            output[f"vq/codebook_loss_{training_sr}"] = codebook_losses[i]
        else:
            for name, loss_fn in (
                ("stft", state.stft_loss),
                ("mel", state.mel_loss),
                ("waveform", state.waveform_loss),
            ):
                per_branch = [
                    loss_fn(recons[i], signal[i]) for i in range(len(working_sr))
                ]
                for i, sr in enumerate(working_sr):
                    output[f"{name}/loss_{training_sr}_{sr}"] = per_branch[i]
                output[f"{name}/loss_{training_sr}"] = sum(per_branch)

            gen_losses, feat_losses = [], []
            for i, sr in enumerate(working_sr):
                gen_i, feat_i = state.gan_loss.generator_loss(recons[i], signal[i], sr)
                output[f"adv/gen_loss_{training_sr}_{sr}"] = gen_i
                output[f"adv/feat_loss_{training_sr}_{sr}"] = feat_i
                gen_losses.append(gen_i)
                feat_losses.append(feat_i)
            output[f"adv/gen_loss_{training_sr}"] = sum(gen_losses)
            output[f"adv/feat_loss_{training_sr}"] = sum(feat_losses)

            # Codebook terms are summed across branches rather than averaged.
            for name, values in (
                ("vq/commitment_loss", commitment_losses),
                ("vq/codebook_loss", codebook_losses),
            ):
                for i, sr in enumerate(working_sr):
                    output[f"{name}_{training_sr}_{sr}"] = values[i]
                output[f"{name}_{training_sr}"] = sum(values[: len(working_sr)])

        output[f"loss_{training_sr}"] = sum(
            v * output[f"{k}_{training_sr}"]
            for k, v in lambdas.items()
            if f"{k}_{training_sr}" in output
        )

    state.optimizer_g.zero_grad()
    accel.backward(output[f"loss_{training_sr}"])
    accel.scaler.unscale_(state.optimizer_g)
    output[f"other/grad_norm_{training_sr}"] = torch.nn.utils.clip_grad_norm_(
        state.generator.parameters(), 1e3
    )
    accel.step(state.optimizer_g)
    state.scheduler_g.step()

    # Skip the AMP scaler update on diverged steps.
    if output[f"loss_{training_sr}"] < 1e3:
        accel.update()

    output[f"other/learning_rate_{training_sr}"] = state.optimizer_g.param_groups[0]["lr"]
    output[f"other/batch_size_{training_sr}"] = signal[0].batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path, training_sr):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", f"mel/loss_{training_sr}"):
        state.tracker.print(f"Best {training_sr} generator so far")
        tags.append(f"best_{training_sr}")
    if state.tracker.step in save_iters:
        tags.append(f"{training_sr}_{state.tracker.step // 1000}k")

    for tag in tags:
        generator_extra = {
            "optimizer.pth": state.optimizer_g.state_dict(),
            "scheduler.pth": state.scheduler_g.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.generator).metadata = metadata
        accel.unwrap(state.generator).save_to_folder(
            f"{save_path}/{tag}", generator_extra
        )

        discriminator_extra = {
            "optimizer.pth": state.optimizer_d.state_dict(),
            "scheduler.pth": state.scheduler_d.state_dict(),
        }
        accel.unwrap(state.discriminator).save_to_folder(
            f"{save_path}/{tag}", discriminator_extra
        )


@torch.no_grad()
def save_samples(state, val_idx, writer, training_sr):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()

    sample_rates = state.generator.sample_rates
    working_sr, i_training_sr = working_sample_rates(sample_rates, training_sr)

    batch = state.val_data.collate([state.val_data[idx] for idx in val_idx])
    batch = util.prepare_batch(batch, accel.device)
    signal = [
        state.val_data.transform(batch[str(sr)].clone(), **batch["transform_args"])
        for sr in working_sr
    ]
    audio_data = [sig.audio_data for sig in signal]

    out = state.generator(audio_data, working_sr, "full")
    recons = [
        AudioSignal(out["audio"][i], signal[i].sample_rate)
        for i in range(len(working_sr))
    ]

    audio_dict = {"recons": recons}
    if state.tracker.step == 0 and i_training_sr is not None:
        audio_dict["signal"] = signal

    for name, signals in audio_dict.items():
        branches = (
            [i_training_sr] if i_training_sr is not None else range(len(signals))
        )
        for i in branches:
            prefix = (
                f"{name}_{sample_rates[i]}"
                if i_training_sr is not None
                else f"{name}_{training_sr}_{sample_rates[i]}"
            )
            for nb in range(signals[i].batch_size):
                signals[i][nb].cpu().write_audio_to_tb(
                    f"{prefix}/sample_{nb}.wav", writer, state.tracker.step
                )


def validate(state, val_dataloader, accel, training_sr):
    for batch in val_dataloader:
        output = val_loop(batch, state, accel, training_sr)
    # Consolidate state dicts if using ZeroRedundancyOptimizer.
    if hasattr(state.optimizer_g, "consolidate_state_dict"):
        state.optimizer_g.consolidate_state_dict()
        state.optimizer_d.consolidate_state_dict()
    return output


def freeze_lower_branches(state, sample_rates, i_training_sr, tracker):
    """Freeze every branch below the one being trained, in all four modules."""
    for i_branch in range(i_training_sr):
        tracker.print(f"Freezing branch {sample_rates[i_branch]} of the model")

        modules = [
            state.generator.encoder.branches[i_branch],
            state.generator.quantizer.RVQs[i_branch],
            state.generator.decoder.decoders[i_branch],
            state.discriminator.DISC[i_branch],
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    save_path: str = "ckpt",
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 72,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
    training_sr: Union[int, str] = 16000,
    lambdas: dict = {
        "mel/loss": 100.0,
        "adv/feat_loss": 2.0,
        "adv/gen_loss": 1.0,
        "vq/commitment_loss": 0.25,
        "vq/codebook_loss": 1.0,
    },
):
    sample_rates = args["DAC.sample_rates"]
    finetuning = training_sr == "finetuning"
    i_training_sr = None if finetuning else sample_rates.index(training_sr)

    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/{training_sr}/logs")
        if accel.local_rank == 0
        else None
    )
    tracker = Tracker(
        writer=writer,
        log_file=f"{save_path}/{training_sr}/log.txt",
        rank=accel.local_rank,
    )

    # Each phase resumes from the best checkpoint of the previous one; only the
    # very first branch starts from scratch.
    if finetuning:
        tracker.print("FINETUNING")
        resume, tag = True, f"best_{sample_rates[-1]}"
    else:
        tracker.print(f"TRAINING FOR {training_sr} RECONSTRUCTION TASK")
        if i_training_sr == 0:
            resume, tag = False, None
        else:
            resume, tag = True, f"best_{sample_rates[i_training_sr - 1]}"

    state = load(args, accel, tracker, save_path, resume, tag=tag)

    train_dataloader = accel.prepare_dataloader(
        state.train_data,
        start_idx=state.tracker.step * batch_size,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=state.train_data.collate,
    )
    train_dataloader = get_infinite_loader(train_dataloader)
    val_dataloader = accel.prepare_dataloader(
        state.val_data,
        start_idx=0,
        num_workers=num_workers,
        batch_size=val_batch_size,
        collate_fn=state.val_data.collate,
        persistent_workers=num_workers > 0,
    )

    # Wrap the loop functions so they track in TensorBoard and progress bars,
    # and so the rank-0-only ones stay rank-0-only.
    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    if finetuning:
        for param in state.generator.parameters():
            param.requires_grad = True
        for param in state.discriminator.parameters():
            param.requires_grad = True
    else:
        freeze_lower_branches(state, sample_rates, i_training_sr, tracker)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            training_label = SEMANTIC_LABELS[
                torch.randint(low=0, high=len(SEMANTIC_LABELS), size=(1,)).item()
            ]

            train_loop(state, batch, accel, lambdas, training_sr, training_label)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer, training_sr)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel, training_sr)
                checkpoint(state, save_iters, save_path, training_sr)
                # Reset the validation bar, print the summary since last time.
                tracker.done("val", f"Iteration {tracker.step}")

            if last_iter:
                break


if __name__ == "__main__":
    args = argbind.parse_args()
    args["args.debug"] = int(os.getenv("LOCAL_RANK", 0)) == 0
    with argbind.scope(args):
        with Accelerator() as accel:
            if accel.local_rank != 0:
                sys.tracebacklimit = 0
            train(args, accel)
