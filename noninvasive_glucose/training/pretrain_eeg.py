"""Pretrain the EEG encoder on synthetic sleep-stage classification."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SLEEP_STAGE_TO_ID = {"deep_sleep": 0, "light_sleep": 1, "awake": 2}


class EEGSleepDataset(Dataset):
    """Turn processed non-invasive windows into EEG sleep-stage examples."""

    def __init__(self, windows: list[dict]):
        self.examples = [
            (
                torch.tensor(window["eeg_bands"], dtype=torch.float32),
                int(SLEEP_STAGE_TO_ID.get(window["sleep_stage_at_t"], 2)),
            )
            for window in windows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, label = self.examples[index]
        return features, torch.tensor(label, dtype=torch.long)


def pretrain_eeg_encoder(
    encoder: nn.Module,
    synthetic_data: list[dict],
    n_epochs: int = 30,
    *,
    config: dict,
    save_path: str | None = None,
) -> dict:
    """Pretrain the EEG encoder to classify the central sleep state."""

    device = config["device"]
    dataset = EEGSleepDataset(synthetic_data)
    loader = DataLoader(
        dataset,
        batch_size=min(int(config["batch_size"]), 64),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )

    head = nn.Linear(int(config["d_model"]), 3).to(device)
    encoder = encoder.to(device)
    optimiser = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    for epoch in range(1, int(n_epochs) + 1):
        encoder.train()
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for eeg_bands, labels in loader:
            eeg_bands = eeg_bands.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pooled = encoder.pooled(eeg_bands)
            logits = head(pooled)
            loss = criterion(logits, labels)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            total_loss += float(loss.item()) * eeg_bands.size(0)
            total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total_examples += int(eeg_bands.size(0))

        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(total_examples, 1),
                "accuracy": total_correct / max(total_examples, 1),
            }
        )

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder_state_dict": encoder.state_dict(), "history": history}, path)

    return {"history": history}


__all__ = ["SLEEP_STAGE_TO_ID", "pretrain_eeg_encoder"]

