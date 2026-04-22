"""Probabilistic prediction head for non-invasive glucose estimation."""

from __future__ import annotations

import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    """Predict both a point estimate and log-variance from the fused CLS token."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, cls_token_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the glucose mean and log-variance in normalised units."""

        raw = self.head(cls_token_output)
        mean = raw[:, 0]
        log_var = raw[:, 1].clamp(-6.0, 6.0)
        return mean, log_var


def nll_loss(mean: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood with predicted heteroscedastic variance."""

    variance = torch.exp(log_var)
    loss = 0.5 * log_var + 0.5 * (target - mean).pow(2) / variance
    return loss.mean()


__all__ = ["UncertaintyHead", "nll_loss"]

