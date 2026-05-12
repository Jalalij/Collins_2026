"""
Seq2Seq GRU Model with Attention for Throughput Prediction
==========================================================
Architecture:
  - Encoder: 2-layer GRU, configurable dropout
  - Decoder: 2-layer GRU, configurable dropout
  - Attention: three 1-layer MLPs, input=128
  - Projection: 2 Linear layers (128->128->1) with LeakyReLU + Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicEmbedding(nn.Module):
    """Encodes cyclic time features (hour, day-of-week) via sin/cos."""

    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.proj = nn.Linear(4, embed_dim)

    def forward(self, hour: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        h_sin = torch.sin(2 * torch.pi * hour / 24)
        h_cos = torch.cos(2 * torch.pi * hour / 24)
        d_sin = torch.sin(2 * torch.pi * dow / 7)
        d_cos = torch.cos(2 * torch.pi * dow / 7)
        x = torch.stack([h_sin, h_cos, d_sin, d_cos], dim=-1)  # (B, T, 4)
        return self.proj(x)                                      # (B, T, embed_dim)


class Attention(nn.Module):
    """Three 1-layer MLPs producing scalar attention weights over encoder outputs."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.query_proj  = nn.Linear(hidden_size, hidden_size)
        self.key_proj    = nn.Linear(hidden_size, hidden_size)
        self.energy_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (B, hidden)
        encoder_outputs: torch.Tensor,  # (B, T_enc, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q       = self.query_proj(decoder_hidden).unsqueeze(1)  # (B, 1, hidden)
        k       = self.key_proj(encoder_outputs)                # (B, T_enc, hidden)
        energy  = self.energy_proj(torch.tanh(q + k))           # (B, T_enc, 1)
        weights = F.softmax(energy.squeeze(-1), dim=-1)         # (B, T_enc)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, hidden)
        return context, weights


class Encoder(nn.Module):
    """2-layer GRU encoder."""

    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,   # applied between GRU layers
        )

    def forward(self, x: torch.Tensor):
        outputs, hidden = self.gru(x)
        return outputs, hidden


class ProjectionHead(nn.Module):
    """2-layer MLP: hidden -> hidden -> 1 with LeakyReLU + Dropout."""

    def __init__(self, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),        # regularise the projection head too
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """2-layer GRU decoder with attention."""

    def __init__(self, num_features: int, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = hidden_size + num_features,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True,
            dropout     = dropout,
        )
        self.attention    = Attention(hidden_size)
        self.projection   = ProjectionHead(hidden_size, dropout)
        self.output_embed = nn.Linear(1, num_features)

    def forward_step(
        self,
        last_output: torch.Tensor,      # (B, 1)
        hidden: torch.Tensor,           # (2, B, hidden)
        encoder_outputs: torch.Tensor,  # (B, T_enc, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        top_hidden             = hidden[-1]
        context, attn_weights  = self.attention(top_hidden, encoder_outputs)
        last_embed             = self.output_embed(last_output)
        gru_input              = torch.cat([context, last_embed], dim=-1).unsqueeze(1)
        gru_out, new_hidden    = self.gru(gru_input, hidden)
        pred                   = self.projection(gru_out.squeeze(1))
        return pred, new_hidden, attn_weights

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_hidden: torch.Tensor,
        output_len: int,
        target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B       = encoder_outputs.size(0)
        hidden  = encoder_hidden
        last_out = torch.zeros(B, 1, device=encoder_outputs.device)

        preds, attns = [], []
        for t in range(output_len):
            pred, hidden, attn = self.forward_step(last_out, hidden, encoder_outputs)
            preds.append(pred)
            attns.append(attn)
            if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                last_out = target[:, t].unsqueeze(1)
            else:
                last_out = pred.detach()

        predictions = torch.cat(preds, dim=-1)       # (B, output_len)
        attn_map    = torch.stack(attns, dim=1)      # (B, output_len, T_enc)
        return predictions, attn_map


class ThroughputSeq2Seq(nn.Module):
    """
    Full Seq2Seq model combining periodic embedding, encoder and decoder.

    Args:
        num_features : number of input feature columns (excluding hour/dow)
        embed_dim    : periodic time embedding dimension
        hidden_size  : GRU hidden size
        dropout      : dropout applied inside GRU layers and projection head
        input_len    : encoder sequence length (default 30)
        output_len   : decoder prediction horizon (default 15)
    """

    def __init__(
        self,
        num_features: int   = 9,
        embed_dim:    int   = 16,
        hidden_size:  int   = 128,
        dropout:      float = 0.3,
        input_len:    int   = 30,
        output_len:   int   = 15,
    ):
        super().__init__()
        self.input_len    = input_len
        self.output_len   = output_len
        self.embed_dim    = embed_dim
        self.num_features = num_features

        self.time_embed        = PeriodicEmbedding(embed_dim)
        encoder_input_size     = num_features + embed_dim
        self.encoder           = Encoder(encoder_input_size, hidden_size, dropout)
        self.decoder           = Decoder(encoder_input_size, hidden_size, dropout)

    def forward(
        self,
        features: torch.Tensor,
        hour:     torch.Tensor,
        dow:      torch.Tensor,
        target:   torch.Tensor | None = None,
        teacher_forcing_ratio: float  = 0.5,
    ):
        time_emb     = self.time_embed(hour.float(), dow.float())
        x            = torch.cat([features, time_emb], dim=-1)
        enc_out, enc_hidden = self.encoder(x)
        predictions, attn_map = self.decoder(
            enc_out, enc_hidden, self.output_len, target, teacher_forcing_ratio
        )
        return predictions, attn_map