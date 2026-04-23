from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """
    Seq2seq LSTM для прогноза спроса.

    Encoder читает историю, decoder читает будущие известные признаки.
    В голову дополнительно подаём последнее значение sales, чтобы модель
    не теряла уровень ряда.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 14,
    ):
        super().__init__()
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x_past)
        dec_out, _ = self.decoder(x_future, (h, c))

        # последнее известное значение sales подаём в голову как якорь уровня
        last_sales = x_past[:, -1:, 0:1].expand(-1, dec_out.size(1), -1)
        y = self.head(torch.cat([dec_out, last_sales], dim=-1)).squeeze(-1)
        return y
