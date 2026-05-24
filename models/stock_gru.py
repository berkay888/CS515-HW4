"""
models/stock_gru.py — StockGRU for Parts (b), (c), and (d).
"""

import torch
import torch.nn as nn


class StockGRU(nn.Module):
    """
    Stacked GRU → Dropout → FC head.

    Architecture mirrors StockLSTM with identical constructor signature so
    the two models are interchangeable in train / test loops.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_dir       = 2 if bidirectional else 1

        # 1-D depthwise conv for auxiliary moving-average features
        self.conv_ma = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=3,
            padding=1,
            groups=input_size,
            bias=False,
        )
        self.conv_act = nn.GELU()
        aug_input = input_size * 2

        self.gru = nn.GRU(
            input_size=aug_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * self.num_dir, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, T, F)
        returns: (batch, output_size)
        """
        x_t  = x.permute(0, 2, 1)
        ma   = self.conv_act(self.conv_ma(x_t))
        ma   = ma.permute(0, 2, 1)
        x_aug = torch.cat([x, ma], dim=-1)

        out, _ = self.gru(x_aug)
        last   = out[:, -1, :]
        last   = self.dropout(last)
        return self.fc(last)
