"""HP-codecX: the audio language model that performs bandwidth extension.

The model mirrors HP-codec's structure: one transformer per RVQ section, so a
harmonic estimator and a percussive estimator run in parallel and never share
weights. Each estimator consumes the low-frequency (16 kHz branch) tokens of
its own section and predicts the high-frequency (48 kHz branch) tokens of that
same section, one codebook at a time.

Prediction proceeds in ``n_pred_tokens`` steps. At step ``k`` the input is the
three low-frequency codebooks plus the ``k`` high-frequency codebooks already
predicted; a learned step embedding tells the network which codebook it is
producing, and a dedicated output head reads it out.

The decoder-only design follows VALL-E (Chen et al., 2025); the section split
is the contribution of this work.
"""

import math

import torch
import torch.nn as nn
from audiotools.ml import BaseModel

SECTIONS = ["harmonic", "percussive"]


class PositionalEncoding(nn.Module):
    r"""Fixed sinusoidal positional encoding.

    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^{2i/d_{model}})
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    dropout : float, optional
        Dropout applied to the sum, by default 0.1.
    max_len : int, optional
        Longest sequence supported, by default 5000.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Registered as a non-trainable parameter so it travels with the
        # checkpoint alongside the rest of the state dict.
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        """Add positional information to ``[T, B, D]`` embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Token embeddings, shape ``[T, B, D]``.
        scale : float
            Multiplier applied to ``x`` before the encoding is added, so that
            embeddings and positions stay on comparable scales.
        """
        x = x * scale
        return self.dropout(x + self.pe[: x.size(0), :])


class TransformerModel(BaseModel):
    """Two parallel transformer estimators, one per RVQ section.

    Parameters
    ----------
    n_input_embs : int
        Number of input token embedding tables per section. Must cover every
        codebook the model can be conditioned on, i.e. the low-frequency
        codebooks plus ``n_pred_tokens - 1`` already-predicted ones.
    ntoken_input : int
        Size of the input codebooks.
    ntoken_input_true : int
        Size of the codebooks as produced by the codec, before any rescaling.
        Kept for configuration compatibility.
    ntoken_output : int
        Size of the predicted codebooks.
    ninp : int
        Model dimension.
    nhead : int
        Number of attention heads.
    nhid : int
        Feed-forward dimension.
    nlayers : int
        Number of transformer encoder layers per section.
    n_pred_tokens : int
        Number of high-frequency codebooks to predict, i.e. prediction steps.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_input_embs: int = 5,
        ntoken_input: int = 1024,
        ntoken_input_true: int = 1024,
        ntoken_output: int = 1024,
        ninp: int = 1024,
        nhead: int = 8,
        nhid: int = 4096,
        nlayers: int = 6,
        n_pred_tokens: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.n_input_embs = n_input_embs
        self.ntoken_input = ntoken_input
        self.ntoken_input_true = ntoken_input_true
        self.ntoken_output = ntoken_output
        self.ninp = ninp
        self.n_pred_tokens = n_pred_tokens

        # `nn.TransformerEncoder` deep-copies the layer it is given, so these
        # prototypes are never actually executed and receive no gradient. They
        # are kept as submodules because the released checkpoints were saved
        # with them in the state dict.
        self.encoder_layers = nn.ModuleDict(
            {
                section: nn.TransformerEncoderLayer(
                    d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout
                )
                for section in SECTIONS
            }
        )
        self.transformers = nn.ModuleDict(
            {
                section: nn.TransformerEncoder(
                    self.encoder_layers[section], num_layers=nlayers
                )
                for section in SECTIONS
            }
        )
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # One embedding table per input codebook; their embeddings are summed,
        # so a token position is represented by the whole residual stack at once.
        self.input_embs = nn.ModuleDict(
            {
                section: nn.ModuleList(
                    nn.Embedding(ntoken_input, ninp) for _ in range(n_input_embs)
                )
                for section in SECTIONS
            }
        )
        # Tells the network which prediction step it is currently on.
        self.step_embs = nn.ModuleDict(
            {section: nn.Embedding(n_pred_tokens, ninp) for section in SECTIONS}
        )
        self.output_heads = nn.ModuleDict(
            {
                section: nn.ModuleList(
                    nn.Linear(ninp, ntoken_output, bias=False)
                    for _ in range(n_pred_tokens)
                )
                for section in SECTIONS
            }
        )

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for section in SECTIONS:
            for i in range(self.n_input_embs):
                nn.init.uniform_(self.input_embs[section][i].weight, -initrange, initrange)
            nn.init.uniform_(self.step_embs[section].weight, -initrange, initrange)
            for i in range(self.n_pred_tokens):
                nn.init.uniform_(
                    self.output_heads[section][i].weight, -initrange, initrange
                )

    def prepare_input(self, src: dict) -> dict:
        """Embed and sum every conditioning codebook of every section.

        Parameters
        ----------
        src : dict
            ``{section: [branch_0, branch_1, ...]}`` where each branch is a
            ``[B, n_codebooks, T]`` tensor of token indices. Branches are
            concatenated along the codebook axis, so embedding table ``q``
            always corresponds to the same (branch, codebook) pair.

        Returns
        -------
        dict
            ``{section: [T, B, ninp]}`` positionally encoded inputs.
        """
        prepared = {}

        for section in src:
            src_sec = src[section]
            B, _, T = src_sec[0].size()
            device = src_sec[0].device

            n_codebooks = sum(branch.size()[1] for branch in src_sec)
            codebooks_per_branch = src_sec[0].size()[1]

            emb_sum = None
            for q in range(n_codebooks):
                branch = src_sec[q // codebooks_per_branch]
                tokens = branch[:, q % codebooks_per_branch, :].long().to(device)
                emb_i = self.input_embs[section][q](tokens)  # [B, T, ninp]
                emb_sum = emb_i if emb_sum is None else emb_sum + emb_i

            if emb_sum is None:
                input_sec = torch.zeros(T, B, self.ninp, device=device)
            else:
                input_sec = emb_sum.permute(1, 0, 2).contiguous()  # [T, B, ninp]

            prepared[section] = self.pos_encoder(input_sec, math.sqrt(self.ninp))

        return prepared

    def forward(self, src: dict, pred_step: torch.Tensor, semantic_label: str) -> dict:
        """Predict one high-frequency codebook per section.

        Parameters
        ----------
        src : dict
            Conditioning tokens, as described in :meth:`prepare_input`.
        pred_step : torch.Tensor
            Index of the codebook being predicted, in ``[0, n_pred_tokens)``.
        semantic_label : str
            ``"full"`` runs both sections; ``"harmonic"`` or ``"percussive"``
            runs that one alone.

        Returns
        -------
        dict
            ``{section: [B, T, ntoken_output]}`` logits.
        """
        sum_src = self.prepare_input(src)
        sections = SECTIONS if semantic_label == "full" else [semantic_label]
        step_idx = int(torch.as_tensor(pred_step).view(-1)[0].item())

        output = {}
        for section in sections:
            device = sum_src[section].device
            step_cond = self.step_embs[section](
                torch.tensor([step_idx], device=device)
            ).view(1, 1, self.ninp)

            enc_output = self.transformers[section](sum_src[section] + step_cond)

            # Each step has its own read-out head. [T, B, V] -> [B, T, V].
            head = self.output_heads[section][step_idx]
            output[section] = torch.transpose(head(enc_output), 0, 1)

        return output
