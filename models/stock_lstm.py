"""
models/stock_lstm.py — StockLSTM for Parts (b), (c), and (d).
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """
    Stacked LSTM → Dropout → FC head.

    Args:
        input_size   : F  — number of input features per time step
        hidden_size  : LSTM hidden dimension
        num_layers   : number of stacked LSTM layers
        output_size  : D = 5 for regression; 1 for binary classification
        dropout      : dropout probability between LSTM layers and before FC
        bidirectional: False for Parts (b)/(c), True for Part (d)
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

        # Optional 1-D conv to generate auxiliary features (moving average)
        self.conv_ma = nn.Conv1d(
            in_channels=input_size,
            out_channels=input_size,
            kernel_size=3,
            padding=1,
            groups=input_size,   # depthwise: one filter per feature
            bias=False,
        )
        self.conv_act = nn.GELU()
        aug_input = input_size * 2          # original + conv-MA features → F̂ = 2F

        self.lstm = nn.LSTM(
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
        # Conv-based moving-average augmentation
        # Conv1d expects (batch, channels, length)
        x_t = x.permute(0, 2, 1)                   # (B, F, T)
        ma   = self.conv_act(self.conv_ma(x_t))     # (B, F, T)
        ma   = ma.permute(0, 2, 1)                  # (B, T, F)
        x_aug = torch.cat([x, ma], dim=-1)          # (B, T, 2F)

        out, _ = self.lstm(x_aug)                   # (B, T, H*dirs)
        last   = out[:, -1, :]                      # (B, H*dirs)
        last   = self.dropout(last)
        return self.fc(last)                        # (B, output_size)
