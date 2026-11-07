"""
models/seq2seq_dist.py - final report distributional Seq2Seq-GRU (MIMO).

Same encoder, attention and residual autoregressive decoder as models/seq2seq.py,
but each target head returns a small predictive distribution instead of one number:

    mean     trained with squared error      -> point forecast for RMSE-type use
    q_0.5    trained with the pinball loss   -> point forecast for MAE-type use
    q_0.025, q_0.25, q_0.75, q_0.975         -> interval / risk forecasts

All outputs are anchored on the last observed value (residual form), and the
quantiles are built as the median plus/minus cumulative softplus gaps, so they
can never cross. The decoder feeds its mean forecast back at the next step.

Why: RTT at 5 Hz is heavy-tailed (RMSE is 3-4x MAE). One squared-error output
is the conditional mean, which is the wrong point forecast for absolute error
and gives no input-dependent uncertainty. Gneiting (2011) shows each scoring
rule has its own optimal point forecast, so the model learns both.

forward() returns (out, attn) where out has shape (B, H, T, 6) ordered as
OUTPUTS below. For code that expects (B, H, T), use .mean_only().
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .seq2seq import BahdanauAttention, Encoder, PeriodicEmbedding

QUANTILES = (0.025, 0.25, 0.5, 0.75, 0.975)
OUTPUTS = ("mean", "q0.025", "q0.25", "q0.5", "q0.75", "q0.975")


class DistHead(nn.Module):
    """Hidden state -> (mean, 5 non-crossing quantiles), all as deltas."""

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Dropout(dropout), nn.Linear(hidden_size, 6),
        )

    def forward(self, h):
        o = self.net(h)
        mean, med = o[:, 0], o[:, 1]
        g = Fn.softplus(o[:, 2:]) + 1e-4          # four positive gaps
        q25 = med - g[:, 0]
        q025 = q25 - g[:, 1]
        q75 = med + g[:, 2]
        q975 = q75 + g[:, 3]
        return torch.stack([mean, q025, q25, med, q75, q975], dim=-1)  # (B, 6)


class DistDecoder(nn.Module):
    def __init__(self, enc_in, hidden_size=128, dropout=0.3, num_targets=1):
        super().__init__()
        self.num_targets = num_targets
        self.gru = nn.GRU(hidden_size + enc_in, hidden_size, num_layers=2,
                          batch_first=True, dropout=dropout)
        self.attention = BahdanauAttention(hidden_size)
        self.heads = nn.ModuleList([DistHead(hidden_size, dropout) for _ in range(num_targets)])
        self.output_embed = nn.Linear(num_targets, enc_in)

    def forward(self, enc_out, enc_hidden, output_len, last_observed, target=None, tf_ratio=0.0):
        hidden = enc_hidden
        last = last_observed
        keys = self.attention.key_proj(enc_out)
        outs, attns = [], []
        for t in range(output_len):
            context, attn = self.attention(hidden[-1], enc_out, keys)
            gru_in = torch.cat([context, self.output_embed(last)], dim=-1).unsqueeze(1)
            h, hidden = self.gru(gru_in, hidden)
            top = h.squeeze(1)
            delta = torch.stack([head(top) for head in self.heads], dim=1)  # (B, T, 6)
            out = last.unsqueeze(-1) + delta
            outs.append(out)
            attns.append(attn)
            mean = out[..., 0]
            if target is not None and tf_ratio > 0:
                m = torch.rand(mean.size(0), 1, device=mean.device) < tf_ratio
                last = torch.where(m, target[:, t, :], mean)
            else:
                last = mean
        return torch.stack(outs, dim=1), torch.stack(attns, dim=1)


class ThroughputSeq2SeqDist(nn.Module):
    def __init__(self, num_features, embed_dim=16, hidden_size=128, dropout=0.3,
                 input_len=30, output_len=15, num_targets=1, target_feature_indices=None):
        super().__init__()
        if target_feature_indices is None:
            raise ValueError("target_feature_indices is required (residual anchor)")
        self.output_len = output_len
        self.num_targets = num_targets
        self.target_feature_indices = list(target_feature_indices)
        self.time_embed = PeriodicEmbedding(embed_dim) if embed_dim > 0 else None
        enc_in = num_features + (embed_dim if embed_dim > 0 else 0)
        self.encoder = Encoder(enc_in, hidden_size, dropout)
        self.decoder = DistDecoder(enc_in, hidden_size, dropout, num_targets)

    def forward(self, features, hour, dow, target=None, teacher_forcing_ratio=0.0):
        x = features
        if self.time_embed is not None:
            x = torch.cat([features, self.time_embed(hour.float(), dow.float())], dim=-1)
        enc_out, enc_hidden = self.encoder(x)
        last = features[:, -1, self.target_feature_indices]
        return self.decoder(enc_out, enc_hidden, self.output_len, last, target, teacher_forcing_ratio)

    @staticmethod
    def mean_only(out):
        return out[..., 0]


def dist_loss(out, y, quantile_weight: float = 1.0):
    """Squared error on the mean + average pinball loss over the quantiles.

    out: (B, H, T, 6), y: (B, H, T), both in normalised units.
    """
    mse = ((out[..., 0] - y) ** 2).mean()
    q = out[..., 1:]
    tau = torch.tensor(QUANTILES, device=out.device, dtype=out.dtype)
    e = y.unsqueeze(-1) - q
    pinball = torch.maximum(tau * e, (tau - 1) * e).mean()
    return mse + quantile_weight * pinball


# ---------------------------------------------------------------------------
# Comparison model: Gaussian (heteroscedastic) heads
# ---------------------------------------------------------------------------
# Same encoder, attention and anchored decoder, but each head predicts a
# normal distribution N(mu, sigma^2) of the change and is trained with the
# Gaussian negative log-likelihood (mean-variance network). Its output uses
# the same 6-channel layout as DistHead, with the quantiles of the normal
# distribution, so the evaluation and calibration code is shared.
_Z = (-1.959964, -0.674490, 0.0, 0.674490, 1.959964)


class GaussHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Dropout(dropout), nn.Linear(hidden_size, 2),
        )

    def forward(self, h):
        o = self.net(h)
        mu = o[:, 0]
        sigma = Fn.softplus(o[:, 1]) + 1e-3
        qs = [mu + z * sigma for z in _Z]
        return torch.stack([mu] + qs, dim=-1)  # (B, 6): mean, q.025, q.25, q.5, q.75, q.975


class ThroughputSeq2SeqGauss(ThroughputSeq2SeqDist):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_size = kwargs.get("hidden_size", 128)
        dropout = kwargs.get("dropout", 0.3)
        self.decoder.heads = nn.ModuleList([GaussHead(hidden_size, dropout)
                                            for _ in range(self.num_targets)])


def gauss_nll(out, y):
    """Gaussian negative log-likelihood (up to a constant); out as above."""
    mu = out[..., 0]
    sigma = (out[..., 5] - out[..., 0]) / _Z[4]
    return (torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2).mean()
