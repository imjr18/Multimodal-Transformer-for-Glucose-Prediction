"""Pretrain the ECG encoder on SDNN reconstruction."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class ECGReconstructionDataset(Dataset):
    """Mask SDNN and ask the encoder to reconstruct it from the other HRV features."""

    def __init__(self, windows: list[dict]):
        self.examples = []
        center_index = 2
        for window in windows:
            features = torch.tensor(window["ecg_features"], dtype=torch.float32)
            masked = features.clone()
            masked[:, 0] = 0.0
            target_sdnn = features[center_index, 0]
            self.examples.append((masked, target_sdnn))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, target = self.examples[index]
        return features, target.unsqueeze(0)


def pretrain_ecg_encoder(
    encoder: nn.Module,
    synthetic_data: list[dict],
    n_epochs: int = 20,
    *,
    config: dict,
    save_path: str | None = None,
) -> dict:
    """Pretrain the ECG encoder to infer SDNN from the remaining HRV features."""

    device = config["device"]
    dataset = ECGReconstructionDataset(synthetic_data)
    loader = DataLoader(
        dataset,
        batch_size=min(int(config["batch_size"]), 64),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )

    head = nn.Linear(int(config["d_model"]), 1).to(device)
    encoder = encoder.to(device)
    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.MSELoss()
    history: list[dict] = []

    for epoch in range(1, int(n_epochs) + 1):
        encoder.train()
        head.train()
        total_loss = 0.0
        total_examples = 0

        for ecg_features, target_sdnn in loader:
            ecg_features = ecg_features.to(device, non_blocking=True)
            target_sdnn = target_sdnn.to(device, non_blocking=True).squeeze(-1)
            pooled = encoder.pooled(ecg_features)
            prediction = head(pooled).squeeze(-1)
            loss = criterion(prediction, target_sdnn)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            total_loss += float(loss.item()) * ecg_features.size(0)
            total_examples += int(ecg_features.size(0))

        history.append({"epoch": epoch, "loss": total_loss / max(total_examples, 1)})

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder_state_dict": encoder.state_dict(), "history": history}, path)

    return {"history": history}


__all__ = ["pretrain_ecg_encoder"]

