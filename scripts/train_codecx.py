"""Train HP-codecX, the bandwidth extension language model.

    python scripts/train_codecx.py --args.load conf/codecx/hpcodecx.yml \
        --save_path runs/hp-codecx

HP-codec is loaded frozen from ``codec_save_path`` and used only to turn audio
into tokens; the gradient flows through the transformers alone.

Each step optimises all ``n_pred_tokens`` prediction stages jointly under
teacher forcing: stage ``k`` is conditioned on the three low-frequency
codebooks plus the ``k`` *ground-truth* high-frequency codebooks, and predicts
the ``k``-th one. The losses of the harmonic and percussive estimators are
summed.

Adapted from the Descript Audio Codec training script (MIT licence).
"""

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import argbind
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
from torcheval.metrics.functional import multiclass_accuracy

sys.path.append(os.getcwd())

import hpcodec  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

torch.backends.cudnn.benchmark = bool(int(os.getenv("CUDNN_BENCHMARK", 1)))

SECTIONS = ["harmonic", "percussive"]

# Optimizer
AdamW = argbind.bind(torch.optim.AdamW, "transformer")
Accelerator = argbind.bind(ml.Accelerator, without_prefix=True)


@argbind.bind("transformer")
def ScheduleLR(
    optimizer,
    wu_start_factor: float = 1.0,
    wu_end_factor: float = 1.7,
    wu_iters: int = 8,
    T_max: int = 1000,
    exp_gamma: float = 1.0,
):
    """Linear warm-up chained into cosine annealing.

    With the defaults in ``conf/codecx/hpcodecx.yml`` the warm-up is disabled
    (zero iterations, unit factors) and only the cosine schedule is active.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, wu_start_factor, wu_end_factor, wu_iters
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    return torch.optim.lr_scheduler.ChainedScheduler(schedulers=[warmup, cosine])


# Models
DAC = argbind.bind(hpcodec.model.DAC)
TransformerModel = argbind.bind(hpcodec.model.TransformerModel)

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

# Losses (used by scripts/evaluate.py, which imports this module).
filter_fn = lambda fn: hasattr(fn, "forward") and "Loss" in fn.__name__
losses = argbind.bind_module(hpcodec.nn.loss, filter_fn=filter_fn)


class MultiDataset(AudioDataset):
    """Serves the same excerpt at every branch sampling rate.

    Unlike the codec's version this one performs no HPSS decomposition: the
    language model is trained on tokens, and the harmonic/percussive split is
    already baked into the frozen codec's RVQ sections.
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
    transformer: TransformerModel

    optimizer_t: AdamW
    scheduler_t: ScheduleLR

    CE_loss: torch.nn.CrossEntropyLoss

    train_data: AudioDataset
    val_data: AudioDataset

    tracker: Tracker


def build_step_inputs(codes: List[dict], n_low_branches: int, step_idx: int):
    """Assemble the conditioning tokens and the target for one prediction stage.

    Parameters
    ----------
    codes : List[dict]
        Codec output, one ``{section: [B, n_codebooks, T]}`` per branch.
    n_low_branches : int
        Number of low-frequency branches used as conditioning; the branch at
        this index is the one being predicted.
    step_idx : int
        Which high-frequency codebook is being predicted.

    Returns
    -------
    input_codes : dict
        ``{section: [low branches..., first step_idx predicted codebooks]}``.
        The trailing entry is absent at ``step_idx == 0``.
    tgt_codes : dict
        ``{section: [B, 1, T]}`` ground-truth tokens for this stage.
    """
    input_codes = {section: [] for section in SECTIONS}
    for i_sr in range(n_low_branches):
        for section in SECTIONS:
            input_codes[section].append(codes[i_sr][section].clone())

    if step_idx > 0:
        # Teacher forcing: condition on the ground-truth codebooks, not on
        # what the model predicted at earlier stages.
        for section in SECTIONS:
            input_codes[section].append(
                codes[n_low_branches][section][:, :step_idx, :].clone()
            )

    tgt_codes = {
        section: codes[n_low_branches][section][:, step_idx, :].unsqueeze(1).clone()
        for section in SECTIONS
    }
    return input_codes, tgt_codes


def prediction_metrics(state, accel, codes, working_sr, compute_diversity: bool):
    """Run every prediction stage and return the per-section losses and metrics."""
    n_tokens_output = accel.unwrap(state.transformer).ntoken_output
    n_pred_tokens = accel.unwrap(state.transformer).n_pred_tokens
    n_low_branches = len(working_sr) - 1

    all_losses = {section: [] for section in SECTIONS}
    all_top1 = {section: [] for section in SECTIONS}
    all_top10 = {section: [] for section in SECTIONS}
    all_diversity = {section: [] for section in SECTIONS}

    for step_idx in range(n_pred_tokens):
        input_codes, tgt_codes = build_step_inputs(codes, n_low_branches, step_idx)

        B, _, T = tgt_codes[SECTIONS[0]].size()
        step_tensor = torch.tensor([step_idx], device=accel.device, dtype=torch.long)
        logits = state.transformer(input_codes, step_tensor, "full")

        flat_logits = {
            section: logits[section].reshape(B * T, n_tokens_output)
            for section in SECTIONS
        }
        flat_targets = {
            section: torch.transpose(tgt_codes[section], -2, -1).reshape(B * T)
            for section in SECTIONS
        }

        for section in SECTIONS:
            all_losses[section].append(
                state.CE_loss(flat_logits[section], flat_targets[section])
            )
            all_top1[section].append(
                multiclass_accuracy(flat_logits[section], flat_targets[section])
            )
            all_top10[section].append(
                multiclass_accuracy(flat_logits[section], flat_targets[section], k=10)
            )

            if compute_diversity:
                # How many distinct tokens the argmax uses, relative to the
                # target's own variety. A collapsing model drives this to zero.
                pred = torch.argmax(logits[section], dim=-1)
                ratio = torch.unique(pred).size(0) / max(
                    torch.unique(tgt_codes[section]).size(0), 1
                )
                all_diversity[section].append(
                    torch.tensor(
                        ratio,
                        device=logits[section].device,
                        dtype=logits[section].dtype,
                    )
                )

    output = {}
    for section in SECTIONS:
        output[f"prediction_loss_{section}"] = torch.mean(torch.stack(all_losses[section]))
        output[f"top_1_accuracy_{section}"] = torch.mean(torch.stack(all_top1[section]))
        output[f"top_10_accuracy_{section}"] = torch.mean(torch.stack(all_top10[section]))
        if compute_diversity:
            output[f"diversity_{section}"] = torch.mean(
                torch.stack(all_diversity[section])
            )

    # The training objective is the sum of the two sections' cross-entropies.
    output["prediction_loss"] = (
        output["prediction_loss_harmonic"] + output["prediction_loss_percussive"]
    )
    for name in ["top_1_accuracy", "top_10_accuracy"] + (
        ["diversity"] if compute_diversity else []
    ):
        output[name] = torch.mean(
            torch.stack([output[f"{name}_{section}"] for section in SECTIONS])
        )

    return output


@argbind.bind(without_prefix=True)
def load(
    args,
    accel: ml.Accelerator,
    tracker: Tracker,
    codec_save_path: str,
    save_path: str,
    resume: bool = False,
    tag: str = "best",
    gen_load_weights: bool = False,
    load_weights: bool = False,
):
    codec_folder = Path(codec_save_path)
    tracker.print(f"Loading frozen codec from {Path('.').absolute()}/{codec_folder}")
    if not (codec_folder / "dac").exists():
        raise FileNotFoundError(
            f"No codec checkpoint at {codec_folder}. `codec_save_path` must point "
            "at the directory containing the `dac/` folder, e.g. "
            "runs/hp-codec/best_finetuning/."
        )
    generator, _ = DAC.load_from_folder(
        folder=str(codec_folder),
        map_location="cpu",
        package=not gen_load_weights,
    )

    tracker.print("Freezing the codec; only the transformers are trained.")
    for param in generator.parameters():
        param.requires_grad = False

    transformer = None
    if resume:
        kwargs = {
            "folder": f"{save_path}/{tag}",
            "map_location": "cpu",
            "package": not load_weights,
        }
        tracker.print(f"Resuming from {Path('.').absolute()}/{kwargs['folder']}")
        if (Path(kwargs["folder"]) / "transformermodel").exists():
            transformer, _ = TransformerModel.load_from_folder(**kwargs)

    transformer = TransformerModel() if transformer is None else transformer
    tracker.print(transformer)

    # The codec is frozen, so it is only moved to the device; the accelerator
    # wraps the transformer alone.
    device = "cpu" if torch.cuda.device_count() == 0 else "cuda"
    generator = generator.to(device)
    transformer = accel.prepare_model(transformer)

    with argbind.scope(args, "transformer"):
        optimizer_t = AdamW(transformer.parameters(), use_zero=accel.use_ddp)
        scheduler_t = ScheduleLR(optimizer_t)

    sample_rates = accel.unwrap(generator).sample_rates
    with argbind.scope(args, "train"):
        train_data = build_dataset(sample_rates)
    with argbind.scope(args, "val"):
        val_data = build_dataset(sample_rates)

    return State(
        generator=generator,
        transformer=transformer,
        optimizer_t=optimizer_t,
        scheduler_t=scheduler_t,
        CE_loss=torch.nn.CrossEntropyLoss(),
        tracker=tracker,
        train_data=train_data,
        val_data=val_data,
    )


def encode_batch(state, batch, accel, inpainting_sr, dataset):
    """Tokenise one batch with the frozen codec."""
    sample_rates = state.generator.sample_rates
    working_sr = [sr for sr in sample_rates if sr <= inpainting_sr]

    signal = [
        dataset.transform(batch[str(sr)].clone(), **batch["transform_args"])
        for sr in working_sr
    ]
    audio_data = [sig.audio_data for sig in signal]

    out = state.generator(audio_data, working_sr, "full")
    return signal, audio_data, working_sr, out["codes"]


@timer()
@torch.no_grad()
def val_loop(batch, state, accel, inpainting_sr, n_pred_tokens):
    state.generator.eval()
    state.transformer.eval()

    batch = util.prepare_batch(batch, accel.device)
    _, _, working_sr, codes = encode_batch(
        state, batch, accel, inpainting_sr, state.val_data
    )

    return prediction_metrics(state, accel, codes, working_sr, compute_diversity=False)


@timer()
def train_loop(state, batch, accel, inpainting_sr, n_pred_tokens):
    state.generator.eval()
    state.transformer.train()

    batch = util.prepare_batch(batch, accel.device)

    with torch.no_grad():
        signal, _, working_sr, codes = encode_batch(
            state, batch, accel, inpainting_sr, state.train_data
        )

    with accel.autocast():
        output = prediction_metrics(
            state, accel, codes, working_sr, compute_diversity=True
        )

    state.optimizer_t.zero_grad()
    accel.backward(output["prediction_loss"])
    accel.scaler.unscale_(state.optimizer_t)
    output["other/grad_norm"] = torch.nn.utils.clip_grad_norm_(
        state.transformer.parameters(), 1e3
    )
    accel.step(state.optimizer_t)
    state.scheduler_t.step()
    accel.update()

    output["other/learning_rate"] = state.optimizer_t.param_groups[0]["lr"]
    output["other/batch_size"] = signal[0].batch_size * accel.world_size

    return {k: v for k, v in sorted(output.items())}


def checkpoint(state, save_iters, save_path):
    metadata = {"logs": state.tracker.history}

    tags = ["latest"]
    state.tracker.print(f"Saving to {str(Path('.').absolute())}")
    if state.tracker.is_best("val", "prediction_loss"):
        state.tracker.print("Best transformer so far")
        tags.append("best")
    if state.tracker.step in save_iters:
        tags.append(f"{state.tracker.step // 1000}k")

    for tag in tags:
        transformer_extra = {
            "optimizer.pth": state.optimizer_t.state_dict(),
            "scheduler.pth": state.scheduler_t.state_dict(),
            "tracker.pth": state.tracker.state_dict(),
            "metadata.pth": metadata,
        }
        accel.unwrap(state.transformer).metadata = metadata
        accel.unwrap(state.transformer).save_to_folder(
            f"{save_path}/{tag}", transformer_extra
        )


def decode_prediction(generator, codes, low_branch_sig, input_sample_rates, lengths):
    """Turn predicted tokens back into a full-band waveform.

    The low branch is passed through untouched: only the high-frequency content
    is synthesised from the predicted tokens, then added to the upsampled input.
    Taking the difference between the decoded branch and its own downsample-then-
    upsample keeps the decoder from re-injecting low frequencies that the input
    already carries.
    """
    latents = generator.quantizer.from_codes(codes, input_sample_rates)

    x = []
    for i_sr in range(len(input_sample_rates)):
        if i_sr == 0:
            x.append(low_branch_sig.audio_data[..., : lengths[i_sr]])
            continue

        decoded = generator.decode([latents[i_sr]], [input_sample_rates[i_sr]])
        recons_sig = AudioSignal(decoded[0], input_sample_rates[i_sr])

        high_freq_sig = (
            recons_sig.clone()
            - recons_sig.clone()
            .resample(input_sample_rates[i_sr - 1])
            .resample(input_sample_rates[i_sr])
        )
        lower_freq_sig = (
            low_branch_sig.clone()
            .resample(input_sample_rates[i_sr])
            .audio_data[..., : lengths[i_sr]]
        )

        x.append(high_freq_sig.audio_data[..., : lengths[i_sr]] + lower_freq_sig)

    return x


def nucleus_sample(probs: torch.Tensor, threshold: float) -> torch.Tensor:
    """Draw one token per position from the smallest top-p mass.

    Keeps the most probable tokens whose cumulative probability stays below
    ``threshold`` (always at least the single most probable one), renormalises,
    and samples. This is what introduces the controlled variability the paper
    reports at inference time.
    """
    ordered, indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(ordered, dim=-1)

    mask = cumulative < threshold
    mask[..., 0] = True

    kept = torch.where(mask, ordered, torch.zeros_like(ordered))
    kept = torch.nn.functional.normalize(kept, p=1, dim=-1)
    probs = torch.zeros_like(probs).scatter(-1, indices, kept)

    flat = probs.view(-1, probs.size(-1))
    return torch.multinomial(flat, num_samples=1).view(probs.size()[:-1])


@torch.no_grad()
def sample_tokens(transformer, codes, n_low_branches, n_pred_tokens, top_p=None):
    """Autoregressively fill in the high-frequency codebooks.

    Unlike training, each stage is conditioned on what the model itself
    predicted at the previous stages.

    Parameters
    ----------
    top_p : float, optional
        Nucleus sampling threshold. ``None`` takes the argmax instead, which is
        what the TensorBoard previews use; the reported results use 0.95.
    """
    input_codes = {section: [] for section in SECTIONS}
    for i_sr in range(n_low_branches):
        for section in SECTIONS:
            input_codes[section].append(codes[i_sr][section].clone())

    for step_idx in range(n_pred_tokens):
        device = input_codes[SECTIONS[0]][0].device
        step_tensor = torch.tensor([step_idx], device=device)
        logits = transformer(input_codes, step_tensor, "full")

        probs = {
            section: torch.nn.functional.softmax(logits[section], dim=-1)
            for section in SECTIONS
        }
        if top_p is None:
            pred = {s: torch.argmax(probs[s], dim=-1) for s in SECTIONS}
        else:
            pred = {s: nucleus_sample(probs[s], top_p) for s in SECTIONS}

        for section in SECTIONS:
            token = torch.unsqueeze(pred[section], dim=1)
            if step_idx == 0:
                input_codes[section].append(token)
            else:
                input_codes[section][-1] = torch.cat(
                    [input_codes[section][-1], token], dim=1
                )

    return [
        {section: input_codes[section][i_sr] for section in SECTIONS}
        for i_sr in range(n_low_branches + 1)
    ]


@torch.no_grad()
def save_samples(state, val_idx, writer, inpainting_sr, n_pred_tokens):
    state.tracker.print("Saving audio samples to TensorBoard")
    state.generator.eval()
    state.transformer.eval()

    batch = state.val_data.collate([state.val_data[idx] for idx in val_idx])
    batch = util.prepare_batch(batch, accel.device)
    signal, audio_data, working_sr, codes = encode_batch(
        state, batch, accel, inpainting_sr, state.val_data
    )
    lengths = [data.size()[-1] for data in audio_data]

    predicted = sample_tokens(
        state.transformer, codes, len(working_sr) - 1, n_pred_tokens
    )
    recons = decode_prediction(
        state.generator, predicted, signal[0], working_sr, lengths
    )
    recons = [
        AudioSignal(recons[i], signal[i].sample_rate) for i in range(len(working_sr))
    ]

    audio_dict = {"recons": recons}
    if state.tracker.step == 0:
        audio_dict["signal"] = signal[: len(working_sr)]

    for name, signals in audio_dict.items():
        for i in range(len(signals)):
            for nb in range(signals[i].batch_size):
                signals[i][nb].cpu().write_audio_to_tb(
                    f"{name}_{working_sr[i]}/sample_{nb}.wav", writer, state.tracker.step
                )


def validate(state, val_dataloader, accel, inpainting_sr, n_pred_tokens):
    outputs = []
    for batch in val_dataloader:
        outputs.append(val_loop(batch, state, accel, inpainting_sr, n_pred_tokens))

    # Consolidate state dicts if using ZeroRedundancyOptimizer.
    if hasattr(state.optimizer_t, "consolidate_state_dict"):
        state.optimizer_t.consolidate_state_dict()

    if not outputs:
        return {}

    keys = set().union(*(out.keys() for out in outputs))
    averaged = {}
    for key in keys:
        values = [out[key] for out in outputs if key in out]
        values = [v if torch.is_tensor(v) else torch.tensor(v) for v in values]
        averaged[key] = torch.stack(values).mean()
    return averaged


@argbind.bind(without_prefix=True)
def train(
    args,
    accel: ml.Accelerator,
    seed: int = 0,
    codec_save_path: str = "ckpt",
    save_path: str = "ckpt",
    inpainting_sr: int = 48000,
    n_pred_tokens: int = 3,
    num_iters: int = 250000,
    save_iters: list = [10000, 50000, 100000, 200000],
    sample_freq: int = 10000,
    valid_freq: int = 1000,
    batch_size: int = 12,
    val_batch_size: int = 10,
    num_workers: int = 8,
    val_idx: list = [0, 1, 2, 3, 4, 5, 6, 7],
):
    Path(save_path).mkdir(exist_ok=True, parents=True)
    writer = (
        SummaryWriter(log_dir=f"{save_path}/logs") if accel.local_rank == 0 else None
    )
    tracker = Tracker(
        writer=writer, log_file=f"{save_path}/log.txt", rank=accel.local_rank
    )

    state = load(args, accel, tracker, codec_save_path, save_path)

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

    global train_loop, val_loop, validate, save_samples, checkpoint
    train_loop = tracker.log("train", "value", history=False)(
        tracker.track("train", num_iters, completed=state.tracker.step)(train_loop)
    )
    val_loop = tracker.track("val", len(val_dataloader))(val_loop)
    validate = tracker.log("val", "mean")(validate)
    save_samples = when(lambda: accel.local_rank == 0)(save_samples)
    checkpoint = when(lambda: accel.local_rank == 0)(checkpoint)

    with tracker.live:
        for tracker.step, batch in enumerate(train_dataloader, start=tracker.step):
            train_loop(state, batch, accel, inpainting_sr, n_pred_tokens)

            last_iter = (
                tracker.step == num_iters - 1 if num_iters is not None else False
            )
            if tracker.step % sample_freq == 0 or last_iter:
                save_samples(state, val_idx, writer, inpainting_sr, n_pred_tokens)

            if tracker.step % valid_freq == 0 or last_iter:
                validate(state, val_dataloader, accel, inpainting_sr, n_pred_tokens)
                checkpoint(state, save_iters, save_path)
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
