"""LSTM baseline for the Part A glucose forecasting task."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """Two-layer LSTM baseline matching the transformer's input and output contract.

    The baseline concatenates heart-rate and glucose-context channels at each
    timestep, encodes them sequentially with a compact two-layer LSTM, and uses
    the final hidden state to predict glucose 30 and 60 minutes ahead.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=2,
            hidden_size=config["lstm_hidden_size"],
            num_layers=config["lstm_num_layers"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(config["lstm_hidden_size"], 64),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(64, len(config["target_offsets"])),
        )

    def forward(self, hr_seq: torch.Tensor, glucose_context: torch.Tensor) -> torch.Tensor:
        """Predict two future glucose targets from concatenated sequential inputs."""

        fused_sequence = torch.cat([hr_seq, glucose_context], dim=-1)
        _, (hidden_state, _) = self.encoder(fused_sequence)
        sequence_summary = hidden_state[-1]
        return self.regression_head(sequence_summary)
