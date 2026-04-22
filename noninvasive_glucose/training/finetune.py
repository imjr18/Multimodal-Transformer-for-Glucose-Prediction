"""Fine-tuning loop for the full non-invasive Transformer."""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from noninvasive_glucose.models.uncertainty_head import nll_loss
from noninvasive_glucose.simulation.noninvasive_simulator import denormalise_glucose


class NonInvasiveWindowDataset(Dataset):
    """Dataset wrapper for processed non-invasive windows."""

    def __init__(self, windows: list[dict]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        return {
            "hr": torch.tensor(window["hr"], dtype=torch.float32),
            "ecg_features": torch.tensor(window["ecg_features"], dtype=torch.float32),
            "emg_features": torch.tensor(window["emg_features"], dtype=torch.float32),
            "eeg_bands": torch.tensor(window["eeg_bands"], dtype=torch.float32),
            "cbf": torch.tensor(window["cbf"], dtype=torch.float32),
            "target": torch.tensor(window["glucose_current"], dtype=torch.float32),
            "user_ids": torch.tensor(window["user_id"], dtype=torch.long),
            "archetype_ids": torch.tensor(window["archetype_id"], dtype=torch.long),
        }

    def get_metadata(self, index: int) -> dict:
        """Expose the original metadata for evaluation routines."""

        return self.windows[index]


class WarmupScheduler:
    """Inverse-square-root scheduler normalised to the configured base learning rate."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.peak_scale = self._schedule_scale(max(warmup_steps, 1))
        self.scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)

    def _schedule_scale(self, step: int) -> float:
        step = max(step, 1)
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def _lr_lambda(self, step: int) -> float:
        adjusted_step = max(step + 1, 1)
        return self._schedule_scale(adjusted_step) / self.peak_scale

    def step(self) -> None:
        self.scheduler.step()

    def state_dict(self) -> dict:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self) -> list[float]:
        return self.scheduler.get_last_lr()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loader(windows: list[dict], config: dict, *, shuffle: bool) -> DataLoader:
    """Create a DataLoader for processed non-invasive windows."""

    return DataLoader(
        NonInvasiveWindowDataset(windows),
        batch_size=int(config["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["num_workers"]),
        pin_memory=bool(config["pin_memory"]),
    )


def _set_pretrained_trainability(model: torch.nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze pretrained encoders."""

    for module in [model.eeg_encoder, model.ecg_encoder]:
        for parameter in module.parameters():
            parameter.requires_grad = trainable


def _create_optimizer(model: torch.nn.Module, config: dict, *, pretrained_unfrozen: bool) -> Adam:
    """Create the optimiser with a lower LR for pretrained encoders after unfreezing."""

    pretrained_params = list(model.eeg_encoder.parameters()) + list(model.ecg_encoder.parameters())
    pretrained_ids = {id(parameter) for parameter in pretrained_params}
    base_params = [parameter for parameter in model.parameters() if id(parameter) not in pretrained_ids and parameter.requires_grad]

    param_groups = [
        {
            "params": base_params,
            "lr": float(config["learning_rate"]),
        }
    ]
    if pretrained_unfrozen:
        param_groups.append(
            {
                "params": [parameter for parameter in pretrained_params if parameter.requires_grad],
                "lr": float(config["pretrained_encoder_lr"]),
            }
        )

    return Adam(
        param_groups,
        betas=tuple(config["adam_betas"]),
        weight_decay=float(config["weight_decay"]),
    )


def load_pretrained_weights(model: torch.nn.Module, config: dict) -> None:
    """Load pretrained EEG and ECG encoder checkpoints when present."""

    eeg_path = Path(config["eeg_pretrain_checkpoint"])
    ecg_path = Path(config["ecg_pretrain_checkpoint"])
    if eeg_path.exists():
        checkpoint = torch.load(eeg_path, map_location="cpu", weights_only=False)
        model.eeg_encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
    if ecg_path.exists():
        checkpoint = torch.load(ecg_path, map_location="cpu", weights_only=False)
        model.ecg_encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=False)


def _collect_predictions(model: torch.nn.Module, loader: DataLoader, *, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Run deterministic validation and collect predictions and targets."""

    model.eval()
    losses: list[float] = []
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            hr = batch["hr"].to(device, non_blocking=True)
            ecg = batch["ecg_features"].to(device, non_blocking=True)
            emg = batch["emg_features"].to(device, non_blocking=True)
            eeg = batch["eeg_bands"].to(device, non_blocking=True)
            cbf = batch["cbf"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            user_ids = batch["user_ids"].to(device, non_blocking=True)
            archetype_ids = batch["archetype_ids"].to(device, non_blocking=True)

            mean, log_var = model(
                hr,
                ecg,
                emg,
                eeg,
                cbf,
                user_ids=user_ids,
                archetype_ids=archetype_ids,
            )
            losses.append(float(nll_loss(mean, log_var, target).item()))
            prediction_batches.append(mean.detach().cpu().numpy())
            target_batches.append(target.detach().cpu().numpy())

    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return predictions, targets, float(np.mean(losses) if losses else 0.0)


def _metrics_from_predictions(predictions: np.ndarray, targets: np.ndarray, norm_stats: dict) -> dict[str, float]:
    """Compute denormalised RMSE and MAE in mg/dL."""

    predictions_mg = denormalise_glucose(predictions, norm_stats)
    targets_mg = denormalise_glucose(targets, norm_stats)
    rmse = float(np.sqrt(np.mean((predictions_mg - targets_mg) ** 2)))
    mae = float(np.mean(np.abs(predictions_mg - targets_mg)))
    return {"rmse": rmse, "mae": mae}


def train_noninvasive_model(
    model: torch.nn.Module,
    train_windows: list[dict],
    val_windows: list[dict],
    norm_stats: dict,
    config: dict,
) -> dict:
    """Fine-tune the full non-invasive model with NLL loss and warmup scheduling."""

    set_seed(int(config["seed"]))
    device = config["device"]
    model = model.to(device)
    model.norm_stats = norm_stats

    train_loader = create_data_loader(train_windows, config, shuffle=True)
    val_loader = create_data_loader(val_windows, config, shuffle=False)

    _set_pretrained_trainability(model, trainable=False)
    optimizer = _create_optimizer(model, config, pretrained_unfrozen=False)
    scheduler = WarmupScheduler(optimizer, d_model=int(config["d_model"]), warmup_steps=int(config["warmup_steps"]))

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, int(config["max_epochs"]) + 1):
        if epoch == int(config["pretrained_freeze_epochs"]) + 1:
            _set_pretrained_trainability(model, trainable=True)
            optimizer = _create_optimizer(model, config, pretrained_unfrozen=True)
            scheduler = WarmupScheduler(optimizer, d_model=int(config["d_model"]), warmup_steps=int(config["warmup_steps"]))

        model.train()
        running_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            hr = batch["hr"].to(device, non_blocking=True)
            ecg = batch["ecg_features"].to(device, non_blocking=True)
            emg = batch["emg_features"].to(device, non_blocking=True)
            eeg = batch["eeg_bands"].to(device, non_blocking=True)
            cbf = batch["cbf"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            user_ids = batch["user_ids"].to(device, non_blocking=True)
            archetype_ids = batch["archetype_ids"].to(device, non_blocking=True)

            mean, log_var = model(
                hr,
                ecg,
                emg,
                eeg,
                cbf,
                user_ids=user_ids,
                archetype_ids=archetype_ids,
            )
            loss = nll_loss(mean, log_var, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["grad_clip"]))
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * hr.size(0)
            total_examples += int(hr.size(0))

        val_predictions, val_targets, val_loss = _collect_predictions(model, val_loader, device=device)
        val_metrics = _metrics_from_predictions(val_predictions, val_targets, norm_stats)

        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(total_examples, 1),
            "val_loss": val_loss,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_record)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | train_loss={epoch_record['train_loss']:.4f} "
                f"| val_loss={val_loss:.4f} | val_rmse={val_metrics['rmse']:.3f} mg/dL"
            )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_path = Path(config["model_checkpoint"])
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "norm_stats": norm_stats,
                    "history": history,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if epochs_without_improvement >= int(config["early_stopping_patience"]):
            break

    Path(config["finetune_history_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["finetune_history_path"]).write_text(json.dumps(history, indent=2), encoding="utf-8")

    return {
        "best_val_rmse": best_val_rmse,
        "best_epoch": best_epoch,
        "history": history,
    }


def load_trained_model(model: torch.nn.Module, checkpoint_path: str, *, device: str) -> dict:
    """Load a saved fine-tuned checkpoint into the provided model instance."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.norm_stats = checkpoint.get("norm_stats")
    return checkpoint


__all__ = [
    "NonInvasiveWindowDataset",
    "WarmupScheduler",
    "create_data_loader",
    "load_pretrained_weights",
    "load_trained_model",
    "train_noninvasive_model",
]

