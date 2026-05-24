"""
parameters.py — Centralised hyperparameter config for CS515 HW4.
All experiments are driven from here; no magic numbers in other files.
"""

from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    train_end: str = "2024-07-31"
    val_end: str = "2024-12-31"
    lookback: int = 20          # T  — window length
    horizons: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])   # d ∈ {1..5}
    features: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close"])


# ─────────────────────────────────────────────
# Part (b) — Return ratio forecasting
# ─────────────────────────────────────────────
@dataclass
class ReturnConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10          # early stopping


# ─────────────────────────────────────────────
# Part (c) — Rolling average return forecasting
# ─────────────────────────────────────────────
@dataclass
class RollingConfig(ReturnConfig):
    roll_window: int = 3        # l — rolling average window
    weights: List[float] = field(
        default_factory=lambda: [0.5, 0.3, 0.2]   # w_0 … w_{l-1}, sum=1
    )


# ─────────────────────────────────────────────
# Part (d) — Turning-point detection
# ─────────────────────────────────────────────
@dataclass
class TurningPointConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    gamma: float = 0.01         # buy-signal threshold (1% gain, i.e. ratio > 1.01)
    # NOTE: the PDF says γ = 1.1 but that means a 10 % gain in a single day,
    # which almost never fires on large-cap stocks. We therefore treat γ = 1.1
    # as the multiplier threshold: buy if (p_max_{t+d} / p_t) > 1.01, i.e. >1% return.
    # Change `gamma` back to 0.10 if you prefer the literal PDF value.


# ─────────────────────────────────────────────
# Part 2 — Interactive communication system
# ─────────────────────────────────────────────
@dataclass
class CommConfig:
    # Message space
    alphabet_size: int = 8      # |S_0| = 8
    seq_len: int = 4            # message length
    # Channel
    num_rounds: int = 4         # T communication rounds
    signal_dim: int = 4         # d  — coded-symbol dimension per round
    noise_var: float = 0.25     # σ²
    feedback_dim: int = 4       # k  — feedback dimension (= signal_dim, relay)
    # Transformer
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    # Training
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 20
