"""Training utilities for Part A models."""

from __future__ import annotations

import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from part_a.evaluate import mae, rmse


class WarmupScheduler:
    """Inverse-square-root scheduler with warmup, rescaled to the configured base LR.

    The original Transformer paper defines the shape of the learning-rate curve.
    In this project we preserve that shape but normalise it so the configured
    `learning_rate` acts as the peak learning rate, keeping training stable for
    the smaller Part A model and dataset.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.peak_scale = self._schedule_scale(max(warmup_steps, 1))
        self.scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)

    def _schedule_scale(self, step: int) -> float:
        step = max(step, 1)
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

    def _lr_lambda(self, step: int) -> float:
        adjusted_step = max(step + 1, 1)
        return self._schedule_scale(adjusted_step) / self.peak_scale

    def step(self) -> None:
        """Advance the wrapped scheduler by one optimisation step."""

        self.scheduler.step()

    def state_dict(self) -> dict:
        """Proxy the wrapped scheduler state for checkpointing."""

        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the wrapped scheduler state from a checkpoint."""

        self.scheduler.load_state_dict(state_dict)

    def get_last_lr(self) -> list[float]:
        """Expose the current learning rate."""

        return self.scheduler.get_last_lr()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Create the Adam optimiser requested in the prompt."""

    return Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=config["adam_betas"],
        weight_decay=config["weight_decay"],
    )


def _validation_metrics(predictions: np.ndarray, targets: np.ndarray, norm_stats: dict) -> dict:
    """Compute horizon-wise validation metrics in denormalised mg/dL units."""

    rmse_30 = rmse(predictions[:, 0], targets[:, 0], norm_stats)
    rmse_60 = rmse(predictions[:, 1], targets[:, 1], norm_stats)
    mae_30 = mae(predictions[:, 0], targets[:, 0], norm_stats)
    mae_60 = mae(predictions[:, 1], targets[:, 1], norm_stats)

    return {
        "val_rmse_30min": rmse_30,
        "val_rmse_60min": rmse_60,
        "val_mae_30min": mae_30,
        "val_mae_60min": mae_60,
        "val_rmse_mean": (rmse_30 + rmse_60) / 2.0,
    }


def _save_json(payload: dict | list, output_path: str | Path) -> None:
    """Save a JSON-serialisable object to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupScheduler,
    checkpoint_path: str | Path,
    *,
    epoch: int,
    best_metric: float,
    config: dict,
    history: list[dict],
) -> None:
    """Persist model, optimiser, scheduler, and history state to disk."""

    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
        "history": history,
    }

    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: WarmupScheduler | None = None,
) -> dict:
    """Load a saved training checkpoint into the supplied objects."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def _run_training_epoch(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupScheduler,
    criterion: nn.Module,
    *,
    device: str,
    grad_clip_norm: float,
) -> float:
    """Run one training epoch and return the average loss."""

    model.train()
    total_loss = 0.0
    total_examples = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for hr_sequence, glucose_context, targets in progress_bar:
        hr_sequence = hr_sequence.to(device, non_blocking=True)
        glucose_context = glucose_context.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(hr_sequence, glucose_context)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        scheduler.step()

        batch_size = hr_sequence.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(total_examples, 1)


def _run_validation_epoch(
    model: torch.nn.Module,
    data_loader,
    criterion: nn.Module,
    *,
    device: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one validation epoch and collect all predictions and targets."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    with torch.no_grad():
        for hr_sequence, glucose_context, targets in data_loader:
            hr_sequence = hr_sequence.to(device, non_blocking=True)
            glucose_context = glucose_context.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            predictions = model(hr_sequence, glucose_context)
            loss = criterion(predictions, targets)

            batch_size = hr_sequence.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            predictions_batches.append(predictions.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())

    average_loss = total_loss / max(total_examples, 1)
    predictions = np.concatenate(predictions_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return average_loss, predictions, targets


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    norm_stats: dict,
    config: dict,
    *,
    checkpoint_path: str | Path,
    history_path: str | Path,
    model_name: str,
) -> dict:
    """Train a model with warmup scheduling and validation-based early stopping."""

    device = config["device"]
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config)
    scheduler = WarmupScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )

    history: list[dict] = []
    best_val_rmse = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = _run_training_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device=device,
            grad_clip_norm=config["grad_clip_norm"],
        )

        val_loss, val_predictions, val_targets = _run_validation_epoch(
            model,
            val_loader,
            criterion,
            device=device,
        )
        validation_metrics = _validation_metrics(val_predictions, val_targets, norm_stats)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **validation_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(epoch_metrics)

        monitored_metric = validation_metrics["val_rmse_mean"]
        if monitored_metric < best_val_rmse:
            best_val_rmse = monitored_metric
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                checkpoint_path,
                epoch=epoch,
                best_metric=best_val_rmse,
                config=config,
                history=history,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % config["progress_print_every"] == 0:
            print(
                f"[{model_name}] epoch {epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_rmse_30={validation_metrics['val_rmse_30min']:.3f} "
                f"val_rmse_60={validation_metrics['val_rmse_60min']:.3f}"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(
                f"[{model_name}] early stopping at epoch {epoch} "
                f"with best mean validation RMSE {best_val_rmse:.3f} mg/dL."
            )
            break

    _save_json(history, history_path)
    return {
        "model_name": model_name,
        "best_val_rmse": best_val_rmse,
        "epochs_trained": len(history),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "history": history,
    }


__all__ = [
    "WarmupScheduler",
    "count_parameters",
    "load_checkpoint",
    "set_seed",
    "train_model",
]
