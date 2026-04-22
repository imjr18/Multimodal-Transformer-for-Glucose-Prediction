"""Dataset utilities for loading processed Part A forecasting windows."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class GlucoseWindowDataset(Dataset):
    """Dataset wrapper around saved sliding-window tensors.

    The processed `.pt` file stores a list of dictionaries, one per forecasting
    window. Each `__getitem__` call returns tensors shaped exactly as the models
    expect: historical heart-rate and glucose context as column vectors and the
    two-horizon glucose target as a length-2 regression target.
    """

    def __init__(self, windows_path: str | Path):
        self.windows_path = Path(windows_path)
        self.windows: list[dict] = torch.load(self.windows_path, map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        """Return the number of available forecasting windows."""

        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one `(hr_sequence, glucose_context, target)` triple."""

        window = self.windows[index]
        hr_sequence = window["hr_input"].to(dtype=torch.float32).unsqueeze(-1)
        glucose_context = window["glucose_input"].to(dtype=torch.float32).unsqueeze(-1)
        target = window["glucose_target"].to(dtype=torch.float32)
        return hr_sequence, glucose_context, target

    def get_metadata(self, index: int) -> dict:
        """Expose window-level metadata without changing the DataLoader API."""

        return self.windows[index]


def create_dataloader(dataset: Dataset, config: dict, *, shuffle: bool) -> DataLoader:
    """Create a DataLoader with the safe worker and pin-memory settings requested."""

    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )
