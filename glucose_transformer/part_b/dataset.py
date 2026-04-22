"""Dataset utilities for Part B multimodal forecasting windows."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class MultiModalWindowDataset(Dataset):
    """Dataset wrapper that exposes HR, glucose context, ECG-HRV, and EMG features.

    Each item returns a fully aligned multimodal forecasting example:
    heart rate and glucose context as scalar sequences, ECG-HRV as five features
    per timestep, EMG as two features per timestep, the two-horizon glucose
    target, and the originating patient ID.
    """

    def __init__(self, windows_path: str | Path):
        self.windows_path = Path(windows_path)
        self.windows: list[dict] = torch.load(self.windows_path, map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        """Return the number of stored multimodal windows."""

        return len(self.windows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return one multimodal forecasting sample."""

        window = self.windows[index]
        hr_sequence = window["hr_input"].to(dtype=torch.float32).unsqueeze(-1)
        glucose_context = window["glucose_input"].to(dtype=torch.float32).unsqueeze(-1)
        ecg_features = window["ecg_features"].to(dtype=torch.float32)
        emg_features = window["emg_features"].to(dtype=torch.float32)
        target = window["glucose_target"].to(dtype=torch.float32)
        patient_id = int(window["patient_id"])
        return hr_sequence, glucose_context, ecg_features, emg_features, target, patient_id

    def get_metadata(self, index: int) -> dict:
        """Expose the raw metadata dictionary for evaluation and plotting."""

        return self.windows[index]
