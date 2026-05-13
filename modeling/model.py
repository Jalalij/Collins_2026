"""
model.py - Seq2Seq GRU with Attention for multi-target throughput prediction.

num_targets == 1  ->  predictions: (B, output_len)            backward-compat
num_targets  > 1  ->  predictions: (B, output_len, num_targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(4, embed_dim)

    def forward(self, hour, dow):
        x = torch.stack([
            torch.sin(2 * torch.pi * hour / 24),
            torch.cos(2 * torch.pi * hour / 24),
            torch.sin(2 * torch.pi * dow  / 7),
            torch.cos(2 * torch.pi * dow  / 7),
        ], dim=-1)
        return self.proj(x)


class Attention(nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.query_proj  = nn.Linear(hidden_size, hidden_size)
        self.key_proj    = nn.Linear(hidden_size, hidden_size)
        self.energy_proj = nn.Linear(hidden_size, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        q       = self.query_proj(decoder_hidden).unsqueeze(1)
        k       = self.key_proj(encoder_outputs)
        weights = F.softmax(self.energy_proj(torch.tanh(q + k)).squeeze(-1), dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2,
                          batch_first=True, dropout=dropout)

    def forward(self, x):
        return self.gru(x)


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Dropout(dropout), nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, num_features, hidden_size=128, dropout=0.3, num_targets=1):
        super().__init__()
        self.num_targets  = num_targets
        self.gru = nn.GRU(hidden_size + num_features, hidden_size,
                          num_layers=2, batch_first=True, dropout=dropout)
        self.attention    = Attention(hidden_size)
        self.heads        = nn.ModuleList([ProjectionHead(hidden_size, dropout)
                                           for _ in range(num_targets)])
        self.output_embed = nn.Linear(num_targets, num_features)

    def _step(self, last_out, hidden, enc_out):
        context, attn = self.attention(hidden[-1], enc_out)
        gru_in        = torch.cat([context, self.output_embed(last_out)], dim=-1).unsqueeze(1)
        h, new_hidden = self.gru(gru_in, hidden)
        pred          = torch.cat([head(h.squeeze(1)) for head in self.heads], dim=-1)
        return pred, new_hidden, attn

    def forward(self, enc_out, enc_hidden, output_len, target=None, tf_ratio=0.5):
        B       = enc_out.size(0)
        hidden  = enc_hidden
        last_out = torch.zeros(B, self.num_targets, device=enc_out.device)
        preds, attns = [], []
        for t in range(output_len):
            pred, hidden, attn = self._step(last_out, hidden, enc_out)
            preds.append(pred); attns.append(attn)
            if target is not None and torch.rand(1).item() < tf_ratio:
                last_out = target[:, t, :] if target.dim() == 3 else target[:, t].unsqueeze(1)
            else:
                last_out = pred.detach()
        predictions = torch.stack(preds, dim=1)          # (B, H, num_targets)
        if self.num_targets == 1:
            predictions = predictions.squeeze(-1)         # (B, H)
        return predictions, torch.stack(attns, dim=1)


class ThroughputSeq2Seq(nn.Module):
    def __init__(self, num_features=11, embed_dim=16, hidden_size=128,
                 dropout=0.3, input_len=30, output_len=15, num_targets=1):
        super().__init__()
        self.input_len   = input_len
        self.output_len  = output_len
        self.num_targets = num_targets
        self.time_embed  = PeriodicEmbedding(embed_dim) if embed_dim > 0 else None
        enc_in           = num_features + (embed_dim if embed_dim > 0 else 0)
        self.encoder     = Encoder(enc_in, hidden_size, dropout)
        self.decoder     = Decoder(enc_in, hidden_size, dropout, num_targets)

    def forward(self, features, hour, dow, target=None, teacher_forcing_ratio=0.5):
        x = features
        if self.time_embed is not None:
            x = torch.cat([features, self.time_embed(hour.float(), dow.float())], dim=-1)
        enc_out, enc_hidden = self.encoder(x)
        return self.decoder(enc_out, enc_hidden, self.output_len, target, teacher_forcing_ratio)
