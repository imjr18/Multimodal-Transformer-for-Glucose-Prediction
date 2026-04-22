"""Training utilities for Part C with gradient accumulation."""

from __future__ import annotations

import gc
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from part_a.evaluate import mae, rmse
from part_a.train import WarmupScheduler, create_optimizer, save_checkpoint


def _save_json(payload: dict | list, output_path: str | Path) -> None:
    """Persist a JSON-serialisable object to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _autocast_context(enabled: bool):
    """Return CUDA autocast when enabled, otherwise a no-op context."""

    if enabled:
        return torch.cuda.amp.autocast()
    return nullcontext()


def _run_training_epoch(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupScheduler,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    *,
    device: str,
    grad_clip_norm: float,
    amp_enabled: bool,
    accumulation_steps: int,
) -> float:
    """Run one training epoch with gradient accumulation."""

    model.train()
    total_loss = 0.0
    total_examples = 0
    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for batch_index, batch in enumerate(progress_bar):
        hr_sequence, glucose_context, ecg_features, emg_features, eeg_signal, cbf_signal, targets, _ = batch
        hr_sequence = hr_sequence.to(device, non_blocking=True)
        glucose_context = glucose_context.to(device, non_blocking=True)
        ecg_features = ecg_features.to(device, non_blocking=True)
        emg_features = emg_features.to(device, non_blocking=True)
        eeg_signal = eeg_signal.to(device, non_blocking=True)
        cbf_signal = cbf_signal.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _autocast_context(amp_enabled):
            predictions = model(
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
            )
            raw_loss = criterion(predictions, targets)
            loss = raw_loss / accumulation_steps

        scaler.scale(loss).backward()

        should_step = ((batch_index + 1) % accumulation_steps == 0) or ((batch_index + 1) == len(data_loader))
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        batch_size = hr_sequence.size(0)
        total_loss += float(raw_loss.item()) * batch_size
        total_examples += batch_size
        progress_bar.set_postfix(loss=f"{raw_loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(total_examples, 1)


def _run_validation_epoch(
    model: torch.nn.Module,
    data_loader,
    criterion: nn.Module,
    *,
    device: str,
    amp_enabled: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one validation epoch and collect all predictions and targets."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            hr_sequence, glucose_context, ecg_features, emg_features, eeg_signal, cbf_signal, targets, _ = batch
            hr_sequence = hr_sequence.to(device, non_blocking=True)
            glucose_context = glucose_context.to(device, non_blocking=True)
            ecg_features = ecg_features.to(device, non_blocking=True)
            emg_features = emg_features.to(device, non_blocking=True)
            eeg_signal = eeg_signal.to(device, non_blocking=True)
            cbf_signal = cbf_signal.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with _autocast_context(amp_enabled):
                predictions = model(
                    hr_sequence,
                    glucose_context,
                    ecg_features,
                    emg_features,
                    eeg_signal,
                    cbf_signal,
                )
                loss = criterion(predictions, targets)

            batch_size = hr_sequence.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            prediction_batches.append(predictions.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())

    average_loss = total_loss / max(total_examples, 1)
    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return average_loss, predictions, targets


def _validation_metrics(predictions: np.ndarray, targets: np.ndarray, norm_stats: dict) -> dict:
    """Compute denormalised validation metrics for both horizons."""

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


def train_full_modal_model(
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
    """Train a Part C model with accumulation, mixed precision, and early stopping."""

    device = config["device"]
    amp_enabled = bool(config["amp_enabled"])
    accumulation_steps = int(config["gradient_accumulation_steps"])
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config)
    scheduler = WarmupScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: list[dict] = []
    best_val_rmse = math.inf
    epochs_without_improvement = 0

    for epoch in range(1, config["max_epochs"] + 1):
        train_loss = _run_training_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            criterion,
            device=device,
            grad_clip_norm=config["grad_clip_norm"],
            amp_enabled=amp_enabled,
            accumulation_steps=accumulation_steps,
        )

        val_loss, val_predictions, val_targets = _run_validation_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            amp_enabled=amp_enabled,
        )
        validation_metrics = _validation_metrics(val_predictions, val_targets, norm_stats)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **validation_metrics,
            "learning_rate": scheduler.get_last_lr()[0],
            "amp_enabled": amp_enabled,
            "gradient_accumulation_steps": accumulation_steps,
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
        "amp_enabled": amp_enabled,
        "gradient_accumulation_steps": accumulation_steps,
        "history": history,
    }


__all__ = ["train_full_modal_model"]
