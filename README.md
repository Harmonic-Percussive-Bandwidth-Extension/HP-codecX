# Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth Extension

Reference implementation of **HP-codec** and **HP-codecX**, the two models
described in *Harmonic-Percussive Disentangled Neural Audio Codec for Bandwidth
Extension* (Giniès, Bie, Fercoq and Richard).

Audio examples, additional architectural detail and pretrained weights:
<https://harmonic-percussive-bandwidth-extension.github.io/>

---

## What the models do

Bandwidth extension is framed here as **token prediction**. Rather than
regressing the missing spectrum directly, the signal is discretised by a codec
whose latent space is arranged so that the missing tokens are predictable from
the observed ones.

**HP-codec** disentangles along two axes at once:

- *Across frequency.* The DAC architecture is replicated into a 16 kHz branch
  (content up to 8 kHz) and a 48 kHz branch (up to 24 kHz). The high branch does
  not encode the signal but the **residual** `s₂₄;₄₈ − ŝ₈;₄₈`, i.e. what the low
  branch could not carry. The two branches are given compression ratios that put
  them at the same token rate, so a low-branch token and a high-branch token
  cover the same temporal context — which is what makes the mapping between them
  positional rather than alignment-dependent.
- *Across structure.* Each branch's RVQ is split into a **harmonic** section and
  a **percussive** section running in parallel on the same encoder output, their
  quantised outputs summed before decoding. Harmonic content (horizontal
  striping along a fundamental and its partials) and percussive content
  (impulse-like, broadband) distribute their energy very differently in
  time-frequency, and separating them gives the predictor a much better-behaved
  target.

The split is enforced by the *training schedule*, not by an auxiliary loss: each
step draws uniformly from `{full, harmonic, percussive}`. A `full` step updates
both sections on unseparated audio; a `harmonic` or `percussive` step updates
only that section, on the corresponding component of an HPSS decomposition.

**HP-codecX** mirrors that structure. Two transformer decoders — one per RVQ
section, no shared weights — read the three 16 kHz codebooks of their own
section and predict the three 48 kHz codebooks of the same section, one at a
time. Each step is conditioned on the codebooks already produced plus a learned
step embedding; training uses teacher forcing, inference uses nucleus sampling
(p = 0.95). The predicted tokens are decoded by the 48 kHz decoder and the
result added back to the upsampled input.

```
              ┌──────────── 16 kHz branch ────────────┐
 s₈;₁₆ ─────► │ encoder ─► harmonic RVQ ─┐            │ ─► ŝ₈;₁₆
              │         └► percussive RVQ ┴─► Σ ─► decoder
              └──────────────┬────────────────────────┘
                             │ tokens                           │ upsample
                             ▼                                  ▼
              harmonic transformer ─┐                           │
              percussive transformer ┴─► Σ ─► 48 kHz decoder ─► + ─► s̃₂₄;₄₈
```

---

## Installation

```bash
git clone <this repository>
cd HP-codecX
pip install -r requirements.txt      # or: pip install -e .
```

Tested with Python 3.11, PyTorch 2.1 and `descript-audiotools` 0.7.2.

ViSQOL is required by `scripts/evaluate.py` only; it comes through
`descript-audiotools` and needs its own binary to be reachable. Every other
script runs without it.

All commands below are run **from the repository root**, which is what puts
`hpcodec/` and `scripts/` on the import path.

---

## Pretrained weights

Download the two checkpoints from the
[latest release](https://github.com/REPLACE-ME/HP-codecX/releases/latest) and
lay them out as the scripts expect:

```
runs/
  hp-codec/best_finetuning/dac/package.pth        # HP-codec       (~1.2 GB)
  hp-codecx/best/transformermodel/package.pth     # HP-codecX LM   (~800 MB)
```

The directory names matter: `load_from_folder` looks for a subfolder named
after the class, so the codec must sit in `dac/` and the language model in
`transformermodel/`. With that in place, the `predict.py`, `reconstruct.py` and
`evaluate.py` commands below run as written.

These are the inference weights. The optimiser states needed to *resume*
training are not distributed; the released checkpoints are enough to reproduce
every reported number, but not to continue training from.

---

## Data

Training uses [MUSDB18](https://sigsep.github.io/datasets/musdb.html) (train
split) and [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) — about
3,800 hours of music in total. Evaluation uses the MUSDB18 test split.

The configuration files ship with placeholder paths. Before training, edit the
`train/`, `val/` and `test/build_dataset.folders` entries in
`conf/codec/16khz.yml` and `conf/codecx/hpcodecx.yml` to point at your own
copies. Every other config inherits from those two.

---

## Training

### 1. HP-codec

Three phases, one command each, all writing into the same directory. Each phase
resumes from the best checkpoint of the previous one; the script handles the
freezing and the checkpoint tags itself.

```bash
# 1. the 16 kHz branch alone
python scripts/train_codec.py --args.load conf/codec/16khz.yml    --save_path runs/hp-codec

# 2. the 48 kHz branch, on the residual, with the 16 kHz branch frozen
python scripts/train_codec.py --args.load conf/codec/48khz.yml    --save_path runs/hp-codec

# 3. both branches unfrozen, jointly finetuned
python scripts/train_codec.py --args.load conf/codec/finetune.yml --save_path runs/hp-codec
```

Checkpoints land in `runs/hp-codec/best_16000`, `best_48000` and
`best_finetuning`. The last one is the codec used everywhere downstream.

### 2. HP-codecX

Set `codec_save_path` in `conf/codecx/hpcodecx.yml` to the finetuned codec
(`runs/hp-codec/best_finetuning/` — the directory *containing* the `dac/`
folder), then:

```bash
python scripts/train_codecx.py --args.load conf/codecx/hpcodecx.yml \
    --save_path runs/hp-codecx
```

The codec is loaded frozen and only supplies tokens; gradients reach the
transformers alone. The `DAC.*` block in the language-model config must match
the one the codec was trained with.

Both models were trained on a single NVIDIA L40S (48 GB): roughly 20 hours for
HP-codec and 27 hours for HP-codecX.

---

## Inference and evaluation

Export a paired test set — each excerpt written once per branch, as
`sample_<i>_sr16000.wav` and `sample_<i>_sr48000.wav`:

```bash
python scripts/save_test_set.py --args.load conf/codec/16khz.yml \
    --output samples/input_16-48
```

Bandwidth extension. Only the 16 kHz file of each pair is read; the 48 kHz one
is the reference that evaluation scores against.

```bash
python scripts/predict.py \
    --path runs/hp-codecx --model_tag best \
    --codec_path runs/hp-codec/best_finetuning \
    --input samples/input_16-48 --output samples/hp-codecx
```

Codec reconstruction, to measure HP-codec on its own:

```bash
python scripts/reconstruct.py \
    --path runs/hp-codec --model_tag best_finetuning \
    --input samples/input_16-48 --output samples/recons_16-48
```

Scoring. Writes `metrics.csv` into the output directory and prints each metric
with a 95% pivotal bootstrap confidence interval:

```bash
# bandwidth extension: score against the full-band reference
python scripts/evaluate.py \
    --input samples/input_16-48 --output samples/hp-codecx --sr_comp 48000

# codec reconstruction: score each branch against its own reference
python scripts/evaluate.py \
    --input samples/input_16-48 --output samples/recons_16-48 \
    --reference per_branch
```

Reported metrics are multi-resolution mel and STFT distances, an ℓ1 waveform
distance, SI-SDR and ViSQOL. The loss objects use their default settings, which
are the ones the paper reports: window lengths {2048, 512} and mel-bin counts
{150, 80}.

Note that SI-SDR is sensitive to sample-level misalignment and so is a poor fit
for synthesis-based systems; it is reported for comparability, not as the
primary criterion.

---

## Repository layout

```
hpcodec/
  model/
    codec.py         HP-codec: branched encoder/decoder, residual coupling
    transformer.py   HP-codecX: the two-section audio language model
    discriminator.py multi-period and multi-band multi-scale STFT, per branch
    base.py          convolution delay / output-length bookkeeping
  nn/
    quantize.py      harmonic + percussive residual vector quantisation
    loss.py          mel, STFT, waveform, SI-SDR, adversarial, feature matching
    layers.py        weight-normalised convolutions, Snake activation
conf/
  codec/             the three HP-codec training phases
  codecx/            HP-codecX
scripts/
  train_codec.py     HP-codec training
  train_codecx.py    HP-codecX training
  save_test_set.py   export a paired test set
  reconstruct.py     codec reconstruction
  predict.py         bandwidth extension
  evaluate.py        objective metrics with bootstrap confidence intervals
```

### A note on names

The codec class is called `DAC` and the language model `TransformerModel`, and
the configuration keys follow (`DAC.*`, `TransformerModel.*`). Checkpoints are
serialised with `torch.package` under the class name, so keeping these names is
what allows the released weights to load. The Python package itself was renamed
from `_dac` to `hpcodec` for clarity; that rename is safe, because
`torch.package` embeds the model source inside the checkpoint.

---

## Scope and limitations

- The pipeline is **two-stage**: HP-codecX cannot be used without the HP-codec
  it was fitted on. Relaxing that coupling — training a language model over an
  off-the-shelf codec's tokens — is the main direction for future work.
- The sampling-rate configuration is **fixed at 16 kHz → 48 kHz**. This follows
  from the reliance on discrete codecs, which operate at fixed rates. Training a
  separate variant for another rate pair is cheap (see the training times above),
  but no single model handles variable input rates.
- The released code covers training, inference and objective evaluation. The
  figure-generating and listening-test analysis scripts are not included.

---

## Acknowledgements and attribution

**This codebase is derived from [Descript Audio
Codec](https://github.com/descriptinc/descript-audio-codec)** (Kumar et al.,
*High-Fidelity Audio Compression with Improved RVQGAN*, NeurIPS 2023), released
under the MIT License. The encoder and decoder blocks, the discriminators, the
loss functions and the overall training-script structure come from that project.
The `VectorQuantize` module is likewise theirs. Individual files carry a note
saying which parts are unchanged.

The contributions of this work are the branched, residually-coupled
architecture, the harmonic/percussive RVQ split and its training schedule, and
the two-section audio language model built on top.

The branched frequency design extends our earlier work, Giniès, Bie, Fercoq and
Richard, *Soft Disentanglement in Frequency Bands for Neural Audio Codecs*,
EUSIPCO 2025.

Harmonic-percussive separation follows Driedger, Müller and Disch, *Extending
Harmonic-Percussive Separation of Audio Signals*, ISMIR 2014, via
`librosa.effects.hpss` with `margin=3.0` and `power=2.0`.

This work was funded by the European Union (ERC, HI-Audio, 101052978). Views and
opinions expressed are those of the authors only and do not necessarily reflect
those of the European Union or the European Research Council. Neither the
European Union nor the granting authority can be held responsible for them.

## Licence

MIT. See [LICENSE](LICENSE).
