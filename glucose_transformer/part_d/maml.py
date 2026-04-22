"""First-order MAML utilities for Part D."""

from __future__ import annotations

import gc
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from part_a.evaluate import mae, rmse


def _slice_task_split(task_split: dict[str, torch.Tensor], size: int | None) -> dict[str, torch.Tensor]:
    """Optionally truncate a support/query split to its first `size` windows."""

    if size is None:
        return task_split
    return {key: value[:size] for key, value in task_split.items()}


def _iter_task_batches(task_split: dict[str, torch.Tensor], batch_size: int):
    """Yield mini-batches from a task split to limit peak VRAM."""

    n_samples = next(iter(task_split.values())).size(0)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield {key: value[start:end] for key, value in task_split.items()}


def _move_split_to_device(task_split: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """Move one support/query split to the requested device."""

    return {
        key: value.to(device, non_blocking=True)
        for key, value in task_split.items()
    }


def compute_task_loss(
    model,
    task_split: dict[str, torch.Tensor],
    *,
    device: str,
    user_id: int,
    archetype_id: int,
    criterion: nn.Module,
    batch_size: int,
    user_embedding_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Compute average loss and collect predictions for one task split."""

    losses: list[torch.Tensor] = []
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    split_on_device = _move_split_to_device(task_split, device)
    for batch in _iter_task_batches(split_on_device, batch_size):
        predictions = model(
            batch["hr_sequence"],
            batch["glucose_context"],
            batch["ecg_features"],
            batch["emg_features"],
            batch["eeg_signal"],
            batch["cbf_signal"],
            user_ids=torch.full((batch["targets"].size(0),), int(user_id), dtype=torch.long, device=device),
            archetype_ids=torch.full((batch["targets"].size(0),), int(archetype_id), dtype=torch.long, device=device),
            user_embedding_override=user_embedding_override,
        )
        losses.append(criterion(predictions, batch["targets"]))
        prediction_batches.append(predictions.detach().cpu().numpy())
        target_batches.append(batch["targets"].detach().cpu().numpy())

    average_loss = torch.stack(losses).mean()
    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    return average_loss, predictions, targets


def adapt_task_model(
    model,
    task: dict,
    config: dict,
    *,
    device: str,
    support_size: int | None = None,
    use_user_embedding_override: bool,
) -> tuple[torch.nn.Module, torch.Tensor | None]:
    """Clone the model and adapt it to one task's support set."""

    task_model = deepcopy(model).to(device)
    criterion = nn.MSELoss()
    task_model.train()

    user_embedding_override = None
    optim_params = list(task_model.parameters())
    if use_user_embedding_override:
        initial_embedding = task_model.get_initial_user_embedding(
            int(task["user_id"]),
            int(task["archetype_id"]),
            device=device,
        )
        user_embedding_override = initial_embedding.detach().clone().requires_grad_(True)
        optim_params = [*optim_params, user_embedding_override]

    inner_optimizer = SGD(optim_params, lr=float(config["maml_inner_lr"]))
    support_split = _slice_task_split(task["support"], support_size)
    support_examples = next(iter(support_split.values())).size(0)

    if support_examples > 0:
        for _ in range(int(config["maml_inner_steps"])):
            inner_optimizer.zero_grad(set_to_none=True)
            support_loss, _, _ = compute_task_loss(
                task_model,
                support_split,
                device=device,
                user_id=int(task["user_id"]),
                archetype_id=int(task["archetype_id"]),
                criterion=criterion,
                batch_size=int(config["batch_size"]),
                user_embedding_override=user_embedding_override,
            )
            support_loss.backward()
            inner_optimizer.step()

    return task_model, user_embedding_override


def evaluate_task_after_adaptation(
    model,
    task: dict,
    norm_stats: dict,
    config: dict,
    *,
    device: str,
    support_size: int,
    use_user_embedding_override: bool = True,
) -> dict:
    """Adapt a model to one user and evaluate the held-out query set."""

    task_model, user_embedding_override = adapt_task_model(
        model,
        task,
        config,
        device=device,
        support_size=support_size,
        use_user_embedding_override=use_user_embedding_override,
    )
    criterion = nn.MSELoss()
    task_model.eval()
    with torch.no_grad():
        query_loss, predictions, targets = compute_task_loss(
            task_model,
            task["query"],
            device=device,
            user_id=int(task["user_id"]),
            archetype_id=int(task["archetype_id"]),
            criterion=criterion,
            batch_size=int(config["batch_size"]),
            user_embedding_override=user_embedding_override,
        )

    metrics = {
        "user_id": int(task["user_id"]),
        "archetype": str(task["archetype"]),
        "support_size": int(support_size),
        "query_loss": float(query_loss.item()),
        "rmse_30min": rmse(predictions[:, 0], targets[:, 0], norm_stats),
        "rmse_60min": rmse(predictions[:, 1], targets[:, 1], norm_stats),
        "mae_30min": mae(predictions[:, 0], targets[:, 0], norm_stats),
        "mae_60min": mae(predictions[:, 1], targets[:, 1], norm_stats),
        "predictions": predictions,
        "targets": targets,
    }

    del task_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


def _accumulate_first_order_gradients(meta_model, task_model, *, scale: float) -> None:
    """Copy gradients from an adapted task model into the meta-model."""

    for meta_param, task_param in zip(meta_model.parameters(), task_model.parameters()):
        if not meta_param.requires_grad or task_param.grad is None:
            continue

        task_grad = task_param.grad.detach().to(meta_param.device) * scale
        if meta_param.grad is None:
            meta_param.grad = task_grad.clone()
        else:
            meta_param.grad.add_(task_grad)


def save_meta_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    *,
    epoch: int,
    best_metric: float,
    history: list[dict],
    config: dict,
) -> None:
    """Save the current Part D meta-learning state."""

    payload = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "config": config,
    }
    checkpoint_file = Path(checkpoint_path)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_file)


def load_meta_checkpoint(model, checkpoint_path: str | Path, *, device: str, optimizer=None) -> dict:
    """Load a Part D meta-learning checkpoint into the supplied objects."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class FOMAML:
    """First-order MAML trainer for the user-conditioned Part D model."""

    def __init__(self, model, dataset, norm_stats: dict, config: dict):
        self.model = model
        self.dataset = dataset
        self.norm_stats = norm_stats
        self.config = config
        self.device = str(config["device"])
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            self.model.parameters(),
            lr=float(config["maml_outer_lr"]),
            betas=config["adam_betas"],
            weight_decay=float(config["weight_decay"]),
        )

    def _validation_metric(self, *, limit: int | None = None) -> tuple[float, list[dict]]:
        """Evaluate adaptation quality on held-out validation users."""

        tasks = self.dataset.get_split_tasks("val", limit=limit)
        rows: list[dict] = []
        for task in tasks:
            metrics = evaluate_task_after_adaptation(
                self.model,
                task,
                self.norm_stats,
                self.config,
                device=self.device,
                support_size=int(self.config["support_set_size"]),
                use_user_embedding_override=True,
            )
            rows.append(metrics)

        mean_rmse = float(np.mean([(row["rmse_30min"] + row["rmse_60min"]) / 2.0 for row in rows])) if rows else math.inf
        return mean_rmse, rows

    def train(
        self,
        *,
        checkpoint_path: str | Path,
        history_path: str | Path,
    ) -> dict:
        """Run the first-order MAML optimisation loop."""

        history: list[dict] = []
        best_val_rmse = math.inf
        epochs_without_improvement = 0
        self.model.to(self.device)
        self.model.set_known_user_ids(self.dataset.get_known_user_ids())

        for epoch in range(1, int(self.config["maml_meta_epochs"]) + 1):
            self.model.train()
            epoch_query_losses: list[float] = []

            for _ in range(int(self.config["meta_steps_per_epoch"])):
                self.optimizer.zero_grad(set_to_none=True)
                tasks = self.dataset.sample_task_batch(split="train", batch_size=int(self.config["meta_batch_size"]))

                for task in tasks:
                    task_model, _ = adapt_task_model(
                        self.model,
                        task,
                        self.config,
                        device=self.device,
                        support_size=int(self.config["support_set_size"]),
                        use_user_embedding_override=False,
                    )
                    task_model.train()
                    query_loss, _, _ = compute_task_loss(
                        task_model,
                        task["query"],
                        device=self.device,
                        user_id=int(task["user_id"]),
                        archetype_id=int(task["archetype_id"]),
                        criterion=self.criterion,
                        batch_size=int(self.config["batch_size"]),
                        user_embedding_override=None,
                    )
                    query_loss.backward()
                    _accumulate_first_order_gradients(
                        self.model,
                        task_model,
                        scale=1.0 / max(len(tasks), 1),
                    )
                    epoch_query_losses.append(float(query_loss.item()))
                    del task_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.config["grad_clip_norm"]))
                self.optimizer.step()
                gc.collect()

            val_rmse, _ = self._validation_metric(limit=int(self.config["meta_val_tasks"]))
            epoch_metrics = {
                "epoch": epoch,
                "train_query_loss": float(np.mean(epoch_query_losses)) if epoch_query_losses else math.inf,
                "val_rmse_mean": val_rmse,
            }
            history.append(epoch_metrics)

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                epochs_without_improvement = 0
                save_meta_checkpoint(
                    self.model,
                    self.optimizer,
                    checkpoint_path,
                    epoch=epoch,
                    best_metric=best_val_rmse,
                    history=history,
                    config=self.config,
                )
            else:
                epochs_without_improvement += 1

            if epoch == 1 or epoch % int(self.config["progress_print_every"]) == 0:
                print(
                    f"[FOMAML] epoch {epoch:03d} "
                    f"train_query_loss={epoch_metrics['train_query_loss']:.4f} "
                    f"val_rmse_mean={val_rmse:.3f}"
                )

            if epochs_without_improvement >= int(self.config["meta_early_stopping_patience"]):
                print(
                    f"[FOMAML] early stopping at epoch {epoch} "
                    f"with best validation RMSE {best_val_rmse:.3f} mg/dL."
                )
                break

        Path(history_path).write_text(json.dumps(history, indent=2), encoding="utf-8")
        return {
            "best_val_rmse": best_val_rmse,
            "epochs_trained": len(history),
            "checkpoint_path": str(checkpoint_path),
            "history_path": str(history_path),
            "history": history,
        }


__all__ = [
    "FOMAML",
    "adapt_task_model",
    "compute_task_loss",
    "evaluate_task_after_adaptation",
    "load_meta_checkpoint",
    "save_meta_checkpoint",
]
