"""
models/comm_system.py — Part 2: Transformer-based interactive AWGN communication.

TX Encoder (Transformer) ──AWGN──► RX Decoder (Transformer)
                   ◄──── noiseless feedback ────
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import CommConfig


# ─────────────────────────────────────────────────────────────────────────────
# Positional encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─────────────────────────────────────────────────────────────────────────────
# TX Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TXEncoder(nn.Module):
    """
    At round t, maps the sequence of 4 symbol-states Z^(t) ∈ R^{4 × d_model}
    to 4 coded symbols x^(t) ∈ R^{4 × signal_dim} satisfying the power constraint.

    Input  per symbol: [original_symbol_embed  |  prev_coded  |  feedback] → MLP → d_model
    Output per symbol: Transformer → MLP → signal_dim, then L2 normalise per batch.
    """

    def __init__(self, cfg: CommConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        A = cfg.alphabet_size
        sig = cfg.signal_dim
        fb  = cfg.feedback_dim

        # Pre-processing MLP: raw-input dim = d_model (symbol embed) + sig + fb
        self.sym_embed = nn.Embedding(A, d)
        self.pre_mlp = nn.Sequential(
            nn.Linear(d + sig + fb, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.pe = PositionalEncoding(d, max_len=cfg.seq_len + 1, dropout=cfg.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_encoder_layers)
        # Post-processing MLP → coded symbol
        self.post_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, sig),
        )

    def forward(
        self,
        symbols: torch.Tensor,           # (B, 4)  integer symbol values
        prev_coded: torch.Tensor,        # (B, 4, signal_dim)  zeros at t=1
        feedback: torch.Tensor,          # (B, 4, feedback_dim) zeros at t=1
    ) -> torch.Tensor:
        """Returns power-constrained coded symbols x ∈ (B, 4, signal_dim)."""
        B = symbols.size(0)
        sym_e = self.sym_embed(symbols)                     # (B, 4, d)
        z = torch.cat([sym_e, prev_coded, feedback], dim=-1)   # (B, 4, d+sig+fb)
        z = self.pre_mlp(z)                                 # (B, 4, d)
        z = self.pe(z)
        h = self.transformer(z)                             # (B, 4, d)
        x = self.post_mlp(h)                                # (B, 4, sig)

        # Power constraint: normalise so ‖x‖² / (4*sig) ≤ 1
        # i.e. scale so mean power per element ≤ 1
        power = x.pow(2).mean(dim=[1, 2], keepdim=True).sqrt() + 1e-8
        x = x / power                                       # satisfies E‖x‖² ≤ 4*sig
        return x


# ─────────────────────────────────────────────────────────────────────────────
# RX Decoder
# ─────────────────────────────────────────────────────────────────────────────

class RXDecoder(nn.Module):
    """
    Executed once after all T rounds.
    Input: all received noisy symbols Y ∈ R^{T × 4 × signal_dim} → estimate m̂.
    """

    def __init__(self, cfg: CommConfig):
        super().__init__()
        self.cfg = cfg
        d   = cfg.d_model
        T   = cfg.num_rounds
        sig = cfg.signal_dim
        A   = cfg.alphabet_size
        L   = cfg.seq_len

        # Pre-processing: flatten T received signals per position → d
        self.pre_mlp = nn.Sequential(
            nn.Linear(T * sig, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.pe = PositionalEncoding(d, max_len=L + 1, dropout=cfg.dropout)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=cfg.num_decoder_layers)
        # Output: logits over alphabet for each of the 4 positions
        self.out_fc = nn.Linear(d, A)

    def forward(self, received_all: torch.Tensor) -> torch.Tensor:
        """
        received_all : (B, T, 4, signal_dim)
        returns      : (B, 4, alphabet_size)  logits
        """
        B, T, L, sig = received_all.shape
        # Reshape: for each of 4 positions, concatenate T received vectors
        y = received_all.permute(0, 2, 1, 3)            # (B, 4, T, sig)
        y = y.reshape(B, L, T * sig)                    # (B, 4, T*sig)
        z = self.pre_mlp(y)                             # (B, 4, d)
        z = self.pe(z)
        h = self.transformer(z)                         # (B, 4, d)
        return self.out_fc(h)                           # (B, 4, A)


# ─────────────────────────────────────────────────────────────────────────────
# Full interactive communication system
# ─────────────────────────────────────────────────────────────────────────────

class InteractiveCommSystem(nn.Module):
    """
    Wraps TXEncoder + RXDecoder for end-to-end training.
    """

    def __init__(self, cfg: CommConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = TXEncoder(cfg)
        self.decoder = RXDecoder(cfg)

    def forward(self, symbols: torch.Tensor) -> torch.Tensor:
        """
        symbols : (B, 4)  integer values in {0 … alphabet_size-1}
        returns : (B, 4, alphabet_size)  logits
        """
        B = symbols.size(0)
        cfg = self.cfg
        T   = cfg.num_rounds
        sig = cfg.signal_dim
        fb  = cfg.feedback_dim
        noise_std = math.sqrt(cfg.noise_var)

        prev_coded = torch.zeros(B, cfg.seq_len, sig, device=symbols.device)
        feedback   = torch.zeros(B, cfg.seq_len, fb,  device=symbols.device)
        received_all = []

        for t in range(T):
            x = self.encoder(symbols, prev_coded, feedback)    # (B, 4, sig)
            # AWGN forward channel
            noise = torch.randn_like(x) * noise_std
            y = x + noise                                      # (B, 4, sig)
            # Noiseless feedback: relay y back to TX (Hint 1)
            prev_coded = x.detach()
            feedback   = y.detach()
            received_all.append(y.unsqueeze(1))               # (B, 1, 4, sig)

        received_all = torch.cat(received_all, dim=1)         # (B, T, 4, sig)
        logits = self.decoder(received_all)                   # (B, 4, A)
        return logits
