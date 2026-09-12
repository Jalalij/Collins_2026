"""
models/seq2seq.py - Improved Seq2Seq GRU with Bahdanau attention (MIMO).

Strategy: MIMO (Multiple Input, Multiple Output)

    Input:
        (B, input_len, num_features + embed_dim)

    Output:
        (B, output_len, num_targets)

The model uses an autoregressive GRU decoder with Bahdanau attention.

===========================================================================
SEQ2SEQ IMPROVEMENTS
===========================================================================

IMPROVEMENT 1:
    Initialize the decoder using the LAST OBSERVED target values instead of
    an all-zero vector.

    OLD:
        last_out = zeros(...)

    NEW:
        last_out = last_observed

    Motivation:
        RTT, jitter, loss, and throughput are temporally correlated.
        The latest observed network state therefore provides a much better
        starting point for short-term forecasting than an arbitrary zero.

IMPROVEMENT 2:
    Remove pred.detach() during autoregressive training.

    OLD:
        last_out = pred.detach()

    NEW:
        last_out = pred

    Motivation:
        detach() prevents gradients from future prediction steps from
        propagating through previous predictions. For a relatively short
        forecast horizon such as 15 seconds, allowing this gradient flow
        can improve multi-step consistency.

IMPROVEMENT 3:
    Use PER-SAMPLE teacher forcing instead of one teacher-forcing decision
    for the entire mini-batch.

    OLD:
        if torch.rand(1).item() < tf_ratio:
            entire batch uses ground truth

    NEW:
        each sample independently decides whether to use:
            - ground truth
            - model prediction

    Motivation:
        This provides a smoother transition between teacher-forced and
        autoregressive training.

IMPROVEMENT 4:
    Use residual/delta forecasting.

    Instead of predicting the next absolute target directly:

        y_hat(t+1)

    the model predicts:

        delta(t+1)

    and constructs:

        y_hat(t+1) = y(t) + delta(t+1)

    Motivation:
        Network KPIs generally have strong short-term temporal persistence.
        Learning changes from the current state is often easier than learning
        the complete absolute value from scratch.

IMPORTANT:
    The targets being predicted must also appear in the encoder feature set,
    because their most recent values are needed to initialize the decoder.

===========================================================================
Batch format from ThroughputFileDataset (io_mode="MIMO")
===========================================================================

    feat : (B, input_len, num_features)
    hour : (B, input_len)
    dow  : (B, input_len)
    tgt  : (B, output_len, num_targets)

All prediction and target tensors remain 3-D.

===========================================================================
Checkpoint statistics
===========================================================================

train.py currently saves model.state_dict().

The feature statistics and target statistics are stored in metadata so that
serve.py can normalize future/live measurements using exactly the same
statistics used during model training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base import BaseTrainer
from train import train as run_training

try:
    from ..dataset import DataConfig, build_dataloaders
except ImportError:
    from dataset import DataConfig, build_dataloaders


# ===========================================================================
# Model components
# ===========================================================================


class PeriodicEmbedding(nn.Module):
    """
    Maps hour-of-day and day-of-week into a learned periodic embedding.

    Instead of passing raw values such as:

        hour = 23
        hour = 0

    which numerically look very different despite being adjacent times,
    the values are first represented using sine/cosine periodic features.

    Inputs
    ------
    hour:
        Shape (B, T), values 0-23.

    dow:
        Shape (B, T), values 0-6.

    Returns
    -------
    Tensor:
        Shape (B, T, embed_dim).
    """

    def __init__(self, embed_dim: int = 16):
        super().__init__()

        # Four periodic variables are generated:
        #
        #   sin(hour)
        #   cos(hour)
        #   sin(day)
        #   cos(day)
        #
        # They are then projected into the requested embedding dimension.
        self.proj = nn.Linear(4, embed_dim)

    def forward(
        self,
        hour: torch.Tensor,
        dow: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.stack(
            [
                torch.sin(2 * torch.pi * hour / 24),
                torch.cos(2 * torch.pi * hour / 24),
                torch.sin(2 * torch.pi * dow / 7),
                torch.cos(2 * torch.pi * dow / 7),
            ],
            dim=-1,
        )

        # Input:
        #   (B, T, 4)
        #
        # Output:
        #   (B, T, embed_dim)
        return self.proj(x)


# ===========================================================================
# Bahdanau attention
# ===========================================================================


class BahdanauAttention(nn.Module):
    """
    Additive/Bahdanau attention over the encoder sequence.

    The decoder uses its current hidden state to determine which historical
    encoder time steps are most relevant for the next prediction.

    Inputs
    ------
    decoder_hidden:
        Shape (B, hidden_size).

    encoder_outputs:
        Shape (B, T, hidden_size).

    Returns
    -------
    context:
        Weighted encoder representation.
        Shape (B, hidden_size).

    weights:
        Attention weights over the historical sequence.
        Shape (B, T).
    """

    def __init__(self, hidden_size: int = 128):
        super().__init__()

        self.query_proj = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.key_proj = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.energy_proj = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Decoder state becomes the attention query.
        #
        # (B, H) -> (B, 1, H)
        q = self.query_proj(
            decoder_hidden
        ).unsqueeze(1)

        # Encoder states become attention keys.
        #
        # (B, T, H) -> (B, T, H)
        k = self.key_proj(
            encoder_outputs
        )

        # Additive/Bahdanau energy.
        #
        # Broadcasting allows:
        #
        #   q: (B, 1, H)
        #   k: (B, T, H)
        #
        # to produce:
        #
        #   energy: (B, T)
        energy = self.energy_proj(
            torch.tanh(q + k)
        ).squeeze(-1)

        # Convert attention energies into probabilities.
        weights = F.softmax(
            energy,
            dim=-1,
        )

        # Weighted sum of the encoder states.
        #
        # weights:
        #   (B, T) -> (B, 1, T)
        #
        # encoder_outputs:
        #   (B, T, H)
        #
        # result:
        #   (B, H)
        context = torch.bmm(
            weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context, weights


# ===========================================================================
# Encoder
# ===========================================================================


class Encoder(nn.Module):
    """
    Two-layer GRU encoder.

    Input
    -----
    x:
        Shape (B, T, enc_in).

    Returns
    -------
    outputs:
        Hidden representation for every historical time step.
        Shape (B, T, hidden_size).

    hidden:
        Final hidden states from both GRU layers.
        Shape (num_layers, B, hidden_size).
    """

    def __init__(
        self,
        enc_in: int,
        hidden_size: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=enc_in,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        return self.gru(x)


# ===========================================================================
# Output projection head
# ===========================================================================


class ProjectionHead(nn.Module):
    """
    Single-target output head.

    Converts one decoder hidden representation into one scalar target.

    Example targets:

        RTT
        jitter
        loss
    """

    def __init__(
        self,
        hidden_size: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.net(x)


# ===========================================================================
# Decoder
# ===========================================================================


class Decoder(nn.Module):
    """
    Autoregressive GRU decoder with Bahdanau attention.

    At each forecasting step:

        1. Compute attention over the historical encoder outputs.

        2. Embed the previous target/prediction.

        3. Concatenate:
               attention context
               previous target embedding

        4. Run one GRU decoder step.

        5. Predict the CHANGE in each target.

        6. Add that change to the previous target value.

    This produces a residual forecasting structure:

        prediction(t+1)
            =
        previous_value(t)
            +
        predicted_change(t+1)

    Output
    ------
    predictions:
        Shape (B, output_len, num_targets).

    attentions:
        Shape (B, output_len, input_len).
    """

    def __init__(
        self,
        enc_in: int,
        hidden_size: int = 128,
        dropout: float = 0.3,
        num_targets: int = 1,
    ):
        super().__init__()

        self.num_targets = num_targets

        # -------------------------------------------------------------------
        # Decoder GRU
        # -------------------------------------------------------------------
        #
        # Decoder input contains:
        #
        #   attention context        -> hidden_size
        #   previous-output embed    -> enc_in
        #
        # Total:
        #
        #   hidden_size + enc_in
        # -------------------------------------------------------------------
        self.gru = nn.GRU(
            input_size=hidden_size + enc_in,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        self.attention = BahdanauAttention(
            hidden_size
        )

        # One projection head per target.
        #
        # For:
        #
        #   RTT
        #   jitter
        #   loss
        #
        # there will be three output heads.
        self.heads = nn.ModuleList(
            [
                ProjectionHead(
                    hidden_size,
                    dropout,
                )
                for _ in range(num_targets)
            ]
        )

        # -------------------------------------------------------------------
        # Previous-output embedding
        # -------------------------------------------------------------------
        #
        # The decoder receives num_targets previous values:
        #
        #   [RTT, jitter, loss]
        #
        # These are projected into enc_in dimensions before being concatenated
        # with the attention context.
        # -------------------------------------------------------------------
        self.output_embed = nn.Linear(
            num_targets,
            enc_in,
        )

    def _step(
        self,
        last_out: torch.Tensor,
        hidden: torch.Tensor,
        enc_out: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Execute one autoregressive decoding step.

        Parameters
        ----------
        last_out:
            Previous target/prediction.

            Shape:
                (B, num_targets)

        hidden:
            Decoder hidden state.

            Shape:
                (num_layers, B, hidden_size)

        enc_out:
            Encoder output sequence.

            Shape:
                (B, input_len, hidden_size)
        """

        # -------------------------------------------------------------------
        # Attention
        # -------------------------------------------------------------------
        #
        # Use the top decoder layer to determine which historical samples
        # should receive the largest attention weights.
        # -------------------------------------------------------------------
        context, attn = self.attention(
            hidden[-1],
            enc_out,
        )

        # Convert previous target values into a decoder feature embedding.
        previous_output_embedding = self.output_embed(
            last_out
        )

        # Concatenate:
        #
        #   attention context
        #   previous output embedding
        #
        # Shape before unsqueeze:
        #
        #   (B, hidden_size + enc_in)
        gru_in = torch.cat(
            [
                context,
                previous_output_embedding,
            ],
            dim=-1,
        ).unsqueeze(1)

        # Run one decoder GRU step.
        #
        # h:
        #   (B, 1, hidden_size)
        #
        # new_hidden:
        #   (num_layers, B, hidden_size)
        h, new_hidden = self.gru(
            gru_in,
            hidden,
        )

        # ================================================================
        # IMPROVEMENT 4 — RESIDUAL / DELTA FORECASTING
        # ================================================================
        #
        # OLD BEHAVIOR:
        #
        #   pred = output_head(hidden)
        #
        # The network had to directly estimate the absolute target value.
        #
        #
        # NEW BEHAVIOR:
        #
        #   delta = output_head(hidden)
        #   pred  = previous_value + delta
        #
        #
        # Example:
        #
        #   previous RTT = 40 ms
        #   network delta prediction = +3 ms
        #
        #   next RTT prediction = 43 ms
        #
        #
        # This is particularly useful for short-horizon network forecasting
        # because RTT, jitter, and loss often change incrementally relative
        # to their latest observed state.
        # ================================================================

        delta = torch.cat(
            [
                head(h.squeeze(1))
                for head in self.heads
            ],
            dim=-1,
        )

        # Shape:
        #
        #   delta:
        #       (B, num_targets)
        #
        #   last_out:
        #       (B, num_targets)
        #
        #   pred:
        #       (B, num_targets)
        pred = last_out + delta

        return pred, new_hidden, attn

    def forward(
        self,
        enc_out: torch.Tensor,
        enc_hidden: torch.Tensor,
        output_len: int,
        last_observed: torch.Tensor,
        target: torch.Tensor | None = None,
        tf_ratio: float = 0.5,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Generate the complete multi-step forecast.

        Parameters
        ----------
        enc_out:
            Encoder outputs.

            Shape:
                (B, input_len, hidden_size)

        enc_hidden:
            Final encoder hidden states.

            Shape:
                (num_layers, B, hidden_size)

        output_len:
            Forecast horizon.

            Example:
                15 means predict t+1 ... t+15.

        last_observed:
            Most recent historical values of the prediction targets.

            Shape:
                (B, num_targets)

        target:
            Future ground-truth sequence used only during training.

            Shape:
                (B, output_len, num_targets)

        tf_ratio:
            Teacher-forcing probability.
        """

        hidden = enc_hidden

        # ================================================================
        # IMPROVEMENT 1 — INITIALIZE WITH LAST OBSERVED TARGETS
        # ================================================================
        #
        # OLD CODE:
        #
        #   last_out = torch.zeros(
        #       B,
        #       self.num_targets,
        #       device=enc_out.device,
        #   )
        #
        #
        # The old decoder therefore started each forecast from:
        #
        #   RTT    = 0
        #   jitter = 0
        #   loss   = 0
        #
        # regardless of the actual network state.
        #
        #
        # NEW CODE:
        #
        # Use the latest measured targets from the historical input window.
        #
        # Example:
        #
        #   last observed:
        #       RTT    = 37 ms
        #       jitter = 4 ms
        #       loss   = 0.5 %
        #
        # becomes the starting state of the decoder.
        #
        # Shape:
        #   (B, num_targets)
        # ================================================================
        last_out = last_observed

        preds = []
        attns = []

        for t in range(output_len):

            # Predict the next target vector.
            pred, hidden, attn = self._step(
                last_out,
                hidden,
                enc_out,
            )

            preds.append(pred)
            attns.append(attn)

            # ============================================================
            # IMPROVEMENT 3 — PER-SAMPLE TEACHER FORCING
            # ============================================================
            #
            # OLD CODE:
            #
            #   if torch.rand(1).item() < tf_ratio:
            #       last_out = target[:, t, :]
            #
            #
            # This makes ONE random decision for the entire mini-batch.
            #
            # For example:
            #
            #   all 512 samples use ground truth
            #
            # OR
            #
            #   all 512 samples use model predictions.
            #
            #
            # NEW CODE:
            #
            # Every sample receives its own teacher-forcing decision.
            #
            # This creates a better mixture of:
            #
            #   ground-truth previous states
            #
            # and
            #
            #   model-generated previous states.
            # ============================================================

            if target is not None and tf_ratio > 0:

                # Generate one independent teacher-forcing decision
                # for every sequence in the batch.
                #
                # Shape:
                #     (B, 1)
                #
                # Broadcasting automatically applies the decision
                # across all target dimensions.
                teacher_mask = (
                    torch.rand(
                        pred.size(0),
                        1,
                        device=pred.device,
                    )
                    < tf_ratio
                )

                # If teacher_mask is True:
                #
                #   use the real target from this forecast step.
                #
                # Otherwise:
                #
                #   use the model's prediction.
                last_out = torch.where(
                    teacher_mask,
                    target[:, t, :],
                    pred,
                )

            else:

                # ========================================================
                # IMPROVEMENT 2 — REMOVE pred.detach()
                # ========================================================
                #
                # OLD:
                #
                #   last_out = pred.detach()
                #
                #
                # detach() stops gradient flow between autoregressive
                # forecast steps.
                #
                #
                # NEW:
                #
                #   last_out = pred
                #
                #
                # This allows errors occurring at later forecast steps
                # to backpropagate through earlier predictions.
                #
                # This is particularly reasonable for your short,
                # 15-step prediction horizon.
                # ========================================================
                last_out = pred

        # Stack decoder predictions:
        #
        # list of output_len tensors:
        #     (B, num_targets)
        #
        # becomes:
        #     (B, output_len, num_targets)
        predictions = torch.stack(
            preds,
            dim=1,
        )

        # Stack attention weights:
        #
        #     (B, output_len, input_len)
        attentions = torch.stack(
            attns,
            dim=1,
        )

        return predictions, attentions


# ===========================================================================
# Full Seq2Seq model
# ===========================================================================


class ThroughputSeq2Seq(nn.Module):
    """
    Complete forecasting model:

        periodic time embedding
                ↓
             encoder
                ↓
         Bahdanau attention
                ↓
       autoregressive decoder
                ↓
        residual predictions

    The output always has shape:

        (B, output_len, num_targets)

    even when num_targets == 1.
    """

    def __init__(
        self,
        num_features: int = 11,
        embed_dim: int = 16,
        hidden_size: int = 128,
        dropout: float = 0.3,
        input_len: int = 30,
        output_len: int = 15,
        num_targets: int = 1,

        # ================================================================
        # IMPROVEMENT 1 SUPPORT
        # ================================================================
        #
        # We must know which encoder feature columns correspond to the
        # prediction targets so that we can extract their LAST OBSERVED
        # values.
        #
        # Example:
        #
        # actual_feature_cols:
        #
        #   [
        #       "rsrp",
        #       "rsrq",
        #       "rtt",
        #       "jitter",
        #       "loss",
        #   ]
        #
        # target_feature_indices:
        #
        #   [2, 3, 4]
        #
        # ================================================================
        target_feature_indices: list[int] | None = None,
    ):
        super().__init__()

        self.output_len = output_len
        self.num_targets = num_targets

        # Save the target positions so the latest historical target
        # measurements can be extracted inside forward().
        self.target_feature_indices = target_feature_indices

        # Optional learned time embedding.
        self.time_embed = (
            PeriodicEmbedding(embed_dim)
            if embed_dim > 0
            else None
        )

        # Encoder input includes:
        #
        #   raw features
        #
        # plus:
        #
        #   learned periodic time embedding.
        enc_in = num_features + (
            embed_dim
            if embed_dim > 0
            else 0
        )

        self.encoder = Encoder(
            enc_in,
            hidden_size,
            dropout,
        )

        self.decoder = Decoder(
            enc_in,
            hidden_size,
            dropout,
            num_targets,
        )

    def forward(
        self,
        features: torch.Tensor,
        hour: torch.Tensor,
        dow: torch.Tensor,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Parameters
        ----------
        features:
            Historical normalized network features.

            Shape:
                (B, input_len, num_features)

        hour:
            Hour-of-day.

            Shape:
                (B, input_len)

        dow:
            Day-of-week.

            Shape:
                (B, input_len)

        target:
            Future target sequence during training.

            Shape:
                (B, output_len, num_targets)

        teacher_forcing_ratio:
            Probability of using ground-truth previous targets during
            decoder training.
        """

        # Begin with the ordinary network features.
        x = features

        # Add periodic hour/day information when enabled.
        if self.time_embed is not None:

            x = torch.cat(
                [
                    features,
                    self.time_embed(
                        hour.float(),
                        dow.float(),
                    ),
                ],
                dim=-1,
            )

        # Encode the historical sequence.
        enc_out, enc_hidden = self.encoder(x)

        # ================================================================
        # IMPROVEMENT 1 — GET THE LATEST OBSERVED TARGET VALUES
        # ================================================================
        #
        # Previously the decoder had no explicit access to the most recent
        # observed RTT/jitter/loss values and therefore started from zeros.
        #
        # target_feature_indices tells us which columns of `features`
        # correspond to the prediction targets.
        # ================================================================

        if self.target_feature_indices is None:
            raise ValueError(
                "target_feature_indices must be provided to ThroughputSeq2Seq. "
                "The improved decoder requires the last observed target values "
                "for initialization."
            )

        # Select:
        #
        #   all batch samples
        #   last historical timestep
        #   only target columns
        #
        # Shape:
        #
        #   (B, num_targets)
        last_observed = features[
            :,
            -1,
            self.target_feature_indices,
        ]

        return self.decoder(
            enc_out=enc_out,
            enc_hidden=enc_hidden,
            output_len=self.output_len,
            last_observed=last_observed,
            target=target,
            tf_ratio=teacher_forcing_ratio,
        )


# ===========================================================================
# Trainer helpers
# ===========================================================================


def _unwrap(ds):
    """
    Recursively descend through ConcatDataset objects until reaching the
    underlying ThroughputFileDataset.
    """

    if hasattr(ds, "datasets"):
        return _unwrap(
            ds.datasets[0]
        )

    return ds


def _serialize_feature_stats(
    feature_stats: dict | None,
) -> dict | None:
    """
    Convert numpy arrays in feature statistics into Python lists.

    metadata.json cannot directly serialize numpy arrays, so this helper
    converts them before returning model metadata.
    """

    if feature_stats is None:
        return None

    return {
        key: (
            value.tolist()
            if hasattr(value, "tolist")
            else value
        )
        for key, value in feature_stats.items()
    }


def _find_target_feature_indices(
    actual_feature_cols: list[str],
    target_cols: list[str],
) -> list[int]:
    """
    Find the input-feature positions corresponding to the prediction targets.

    This helper is required by IMPROVEMENT 1.

    The improved decoder starts from the most recent observed target values.
    Therefore each target must also appear somewhere in the historical
    encoder input.

    Example
    -------

    actual_feature_cols:

        [
            "rsrp",
            "rsrq",
            "rtt",
            "jitter",
            "loss",
        ]

    target_cols:

        [
            "rtt",
            "jitter",
            "loss",
        ]

    Result:

        [2, 3, 4]

    Raises
    ------
    ValueError
        If a prediction target does not appear among the historical
        input features.
    """

    target_feature_indices = []

    for target_col in target_cols:

        if target_col not in actual_feature_cols:

            raise ValueError(
                "\nSeq2Seq residual decoder configuration error:\n"
                f"  Target '{target_col}' was not found in actual_feature_cols.\n\n"
                "The improved Seq2Seq decoder uses the last observed value of "
                "each target to initialize forecasting. Therefore every target "
                "must also be included as an encoder input feature.\n\n"
                f"Targets:\n  {target_cols}\n\n"
                f"Available features:\n  {actual_feature_cols}\n"
            )

        target_feature_indices.append(
            actual_feature_cols.index(
                target_col
            )
        )

    return target_feature_indices


# ===========================================================================
# Seq2Seq trainer
# ===========================================================================


class Seq2SeqTrainer(BaseTrainer):
    """
    MIMO trainer for ThroughputSeq2Seq.

    API:

        train_and_evaluate(
            cfg,
            results_dir,
            config,
        )

    Output dictionary keys:

    Single target:

        t+1s
        t+2s
        ...

    Multiple targets:

        <target>_t+1s
        <target>_t+2s
        ...
    """

    @staticmethod
    def _short(
        col: str,
    ) -> str:
        """
        Produce shorter names for result dictionary keys.
        """

        return (
            col.replace(
                "dish_",
                "",
            )
            .replace(
                "_throughput_mbps",
                "",
            )
            .replace(
                "_ms_mean",
                "_ms",
            )
            .replace(
                "_mbps",
                "",
            )
        )

    @staticmethod
    def train_and_evaluate(
        cfg: DataConfig,
        results_dir: str,
        config: dict,
    ) -> dict:

        print(
            "\n"
            + "=" * 20
            + " SEQ2SEQ MODEL (MIMO) "
            + "=" * 20
        )

        # ===================================================================
        # Hyperparameters
        # ===================================================================

        hidden_size = config.get(
            "hidden_size",
            128,
        )

        embed_dim = config.get(
            "embed_dim",
            16,
        )

        dropout = config.get(
            "dropout",
            0.3,
        )

        epochs = config.get(
            "epochs",
            100,
        )

        lr = config.get(
            "lr",
            1e-3,
        )

        batch_size = config.get(
            "batch_size",
            512,
        )

        weight_decay = config.get(
            "weight_decay",
            1e-3,
        )

        es_patience = config.get(
            "early_stop_patience",
            40,
        )

        ema_alpha = config.get(
            "ema_alpha",
            0.3,
        )

        # ===================================================================
        # Data
        # ===================================================================

        # feature_stats and target_stats are retained because live inference
        # must normalize measurements using the SAME statistics used during
        # training.
        (
            train_loader,
            val_loader,
            test_loader,
            feature_stats,
            target_stats,
            actual_feature_cols,
        ) = build_dataloaders(cfg)

        num_features = len(
            actual_feature_cols
        )

        num_targets = cfg.num_targets
        output_len = cfg.output_len
        target_cols = cfg.targets

        print(
            f"  num_features : {num_features} "
            "(actual after column intersection)"
        )

        print(
            f"  num_targets  : {num_targets} "
            f"-> {target_cols}"
        )

        print(
            f"  output_len   : {output_len}"
        )

        # ===================================================================
        # Verify the target tensor shape
        # ===================================================================

        feat_b, _, _, tgt_b = next(
            iter(train_loader)
        )

        assert tgt_b.shape == (
            tgt_b.shape[0],
            output_len,
            num_targets,
        ), (
            f"Expected target shape "
            f"(B, {output_len}, {num_targets}), "
            f"got {tuple(tgt_b.shape)}. "
            f"Ensure DataConfig.io_mode='MIMO'."
        )

        # ===================================================================
        # IMPROVEMENT 1 — Locate target columns in the historical features
        # ===================================================================
        #
        # The decoder must know where RTT/jitter/loss are located in the
        # encoder feature tensor so it can retrieve their latest measurements.
        #
        # We determine the positions dynamically rather than hardcoding them.
        #
        # This is important because Starlink, OneWeb, and Quectel may have
        # different feature layouts.
        # ===================================================================

        target_feature_indices = _find_target_feature_indices(
            actual_feature_cols=actual_feature_cols,
            target_cols=target_cols,
        )

        print(
            f"  target feature indices : "
            f"{target_feature_indices}"
        )

        for target_name, target_index in zip(
            target_cols,
            target_feature_indices,
        ):
            print(
                f"    {target_name} -> "
                f"features[:, :, {target_index}]"
            )

        # ===================================================================
        # Model
        # ===================================================================

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = ThroughputSeq2Seq(
            num_features=num_features,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            dropout=dropout,
            input_len=feat_b.shape[1],
            output_len=output_len,
            num_targets=num_targets,

            # ===============================================================
            # IMPROVEMENT 1
            # ===============================================================
            # Pass the dynamically discovered target positions into the model
            # so the final historical target values can initialize the decoder.
            # ===============================================================
            target_feature_indices=target_feature_indices,
        ).to(device)

        # ===================================================================
        # Training
        # ===================================================================

        # If no validation split exists, retain the previous fallback behavior
        # and use the test loader as effective validation data.
        effective_val = (
            val_loader
            if val_loader is not None
            else test_loader
        )

        history = run_training(
            model,
            train_loader,
            effective_val,
            test_loader=test_loader,

            # Explicitly state MIMO mode rather than relying on train.py's
            # default behavior.
            mode="mimo",

            num_epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            early_stop_patience=es_patience,
            ema_alpha=ema_alpha,
            save_dir=results_dir,
        )

        # ===================================================================
        # Test inference
        # ===================================================================

        base_ds = _unwrap(
            train_loader.dataset
        )

        model.eval()

        preds_list = []
        true_list = []

        with torch.no_grad():

            for feat, hour, dow, tgt in test_loader:

                # No target tensor is provided here.
                #
                # Therefore teacher forcing is automatically disabled and the
                # decoder operates fully autoregressively during evaluation.
                preds, _ = model(
                    feat.to(device),
                    hour.to(device),
                    dow.to(device),
                )

                preds_list.append(
                    preds.cpu()
                )

                true_list.append(
                    tgt
                )

        # Concatenate batches.
        #
        # Both resulting tensors have shape:
        #
        #   (N, output_len, num_targets)
        preds_arr = torch.cat(
            preds_list,
            dim=0,
        )

        targets_arr = torch.cat(
            true_list,
            dim=0,
        )

        # ===================================================================
        # Denormalization
        # ===================================================================

        # inverse_transform_all expects target vectors. We therefore apply it
        # independently at every forecast horizon.
        preds_denorm = np.stack(
            [
                base_ds
                .inverse_transform_all(
                    preds_arr[:, t, :]
                )
                .numpy()
                for t in range(output_len)
            ],
            axis=1,
        )

        true_denorm = np.stack(
            [
                base_ds
                .inverse_transform_all(
                    targets_arr[:, t, :].float()
                )
                .numpy()
                for t in range(output_len)
            ],
            axis=1,
        )

        # ===================================================================
        # Build evaluation dictionaries
        # ===================================================================

        y_true_dict: dict[
            str,
            np.ndarray,
        ] = {}

        y_pred_dict: dict[
            str,
            np.ndarray,
        ] = {}

        if num_targets == 1:

            for h in range(output_len):

                key = f"t+{h + 1}s"

                y_true_dict[key] = (
                    true_denorm[
                        :,
                        h,
                        0,
                    ]
                )

                y_pred_dict[key] = (
                    preds_denorm[
                        :,
                        h,
                        0,
                    ]
                )

        else:

            for ti, col in enumerate(
                target_cols
            ):

                short = Seq2SeqTrainer._short(
                    col
                )

                for h in range(output_len):

                    key = (
                        f"{short}_t+{h + 1}s"
                    )

                    y_true_dict[key] = (
                        true_denorm[
                            :,
                            h,
                            ti,
                        ]
                    )

                    y_pred_dict[key] = (
                        preds_denorm[
                            :,
                            h,
                            ti,
                        ]
                    )

        # ===================================================================
        # Return model results and metadata
        # ===================================================================

        return {
            "model_name": "Seq2Seq-GRU",

            "y_true_dict": y_true_dict,
            "y_pred_dict": y_pred_dict,

            "metadata": {

                "model_type": "Seq2Seq-GRU",

                # MIMO:
                # predict the complete multi-step output trajectory jointly.
                "strategy": "MIMO",

                "hidden_size": hidden_size,
                "embed_dim": embed_dim,
                "num_features": num_features,
                "num_targets": num_targets,
                "target_cols": target_cols,
                "output_len": output_len,
                "input_len": feat_b.shape[1],

                "epochs_trained": len(
                    history["train_loss"]
                ),

                "checkpoint": history.get(
                    "checkpoint",
                    "",
                ),

                "parameters": sum(
                    p.numel()
                    for p in model.parameters()
                ),

                # ===========================================================
                # IMPROVEMENT METADATA
                # ===========================================================
                #
                # Store the decoder configuration so future experiments can
                # identify that this checkpoint uses the improved Seq2Seq
                # formulation.
                # ===========================================================

                "decoder_initialization": "last_observed",

                "residual_prediction": True,

                "per_sample_teacher_forcing": True,

                "autoregressive_gradient_detach": False,

                "target_feature_indices": target_feature_indices,

                # ===========================================================
                # Live-inference metadata
                # ===========================================================

                # Actual features after dataset column intersection.
                "actual_feature_cols": actual_feature_cols,

                # Training-time feature normalization statistics.
                "feature_stats": _serialize_feature_stats(
                    feature_stats
                ),

                # Training-time target normalization statistics.
                "target_stats": target_stats,

                "normalise": cfg.normalise,
            },
        }