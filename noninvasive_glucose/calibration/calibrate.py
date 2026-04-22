"""Deployment-time user calibration for the non-invasive estimator."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch

from noninvasive_glucose.models.uncertainty_head import nll_loss


def _normalise_glucose_scalar(glucose_value: float, norm_stats: dict) -> float:
    """Normalise one scalar glucose reading using training statistics."""

    stats = norm_stats["glucose_current"]
    return float((glucose_value - float(stats["mean"])) / float(stats["std"]))


def _to_biosignal_tensors(biosignals: dict[str, Any], *, device: str) -> dict[str, torch.Tensor]:
    """Convert one calibration biosignal window into model-ready tensors."""

    def _tensor(value, *, dtype=torch.float32):
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=dtype)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device=device, dtype=dtype)

    return {
        "hr": _tensor(biosignals["hr"]),
        "ecg_features": _tensor(biosignals["ecg_features"]),
        "emg_features": _tensor(biosignals["emg_features"]),
        "eeg_bands": _tensor(biosignals["eeg_bands"]),
        "cbf": _tensor(biosignals["cbf"]),
        "user_ids": torch.tensor([int(biosignals.get("user_id", 0))], dtype=torch.long, device=device),
        "archetype_ids": torch.tensor([int(biosignals.get("archetype_id", 0))], dtype=torch.long, device=device),
    }


class UserCalibrator:
    """Personalise a trained model using a few sparse calibration readings.

    Only the user embedding is adapted. All shared modality encoders and fusion
    layers stay frozen so the model preserves its learned physiological prior.
    """

    def __init__(self, model: torch.nn.Module, config: dict, norm_stats: dict | None = None):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.norm_stats = norm_stats if norm_stats is not None else getattr(model, "norm_stats", None)
        if self.norm_stats is None:
            raise ValueError("UserCalibrator requires normalisation statistics.")

    def calibrate(self, calibration_pairs: list[tuple[dict[str, Any], float]]) -> torch.nn.Module:
        """Adapt a user embedding with a small MAML-style inner loop."""

        model_copy = deepcopy(self.model).to(self.device)
        for parameter in model_copy.parameters():
            parameter.requires_grad = False

        first_biosignals, _ = calibration_pairs[0]
        archetype_id = int(first_biosignals.get("archetype_id", 0))
        initial_embedding = model_copy.user_embeddings.resolve(
            batch_size=1,
            user_ids=None,
            archetype_ids=torch.tensor([archetype_id], dtype=torch.long, device=self.device),
            device=self.device,
        ).detach()
        adapted_embedding = torch.nn.Parameter(initial_embedding.clone())

        optimiser = torch.optim.SGD([adapted_embedding], lr=float(self.config["calibration_inner_lr"]))

        for _ in range(int(self.config["calibration_inner_steps"])):
            for biosignals, glucose_value in calibration_pairs:
                tensors = _to_biosignal_tensors(biosignals, device=self.device)
                target = torch.tensor(
                    [_normalise_glucose_scalar(float(glucose_value), self.norm_stats)],
                    dtype=torch.float32,
                    device=self.device,
                )
                mean, log_var = model_copy(
                    tensors["hr"],
                    tensors["ecg_features"],
                    tensors["emg_features"],
                    tensors["eeg_bands"],
                    tensors["cbf"],
                    user_ids=tensors["user_ids"],
                    archetype_ids=tensors["archetype_ids"],
                    user_embedding_override=adapted_embedding,
                )
                loss = nll_loss(mean, log_var, target)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

        model_copy.set_calibration_embedding(adapted_embedding.detach())
        model_copy.norm_stats = self.norm_stats
        return model_copy


__all__ = ["UserCalibrator"]

