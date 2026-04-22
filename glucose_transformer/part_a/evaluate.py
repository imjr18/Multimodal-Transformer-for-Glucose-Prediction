"""Evaluation utilities for Part A glucose forecasting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _extract_glucose_stats(norm_stats: dict) -> dict[str, float]:
    """Return the glucose normalisation statistics from the saved stats dictionary."""

    if "glucose_mg_dl" in norm_stats:
        return norm_stats["glucose_mg_dl"]
    if "glucose" in norm_stats:
        return norm_stats["glucose"]
    raise KeyError("Could not find glucose statistics in the normalisation dictionary.")


def denormalise_glucose(values: np.ndarray | torch.Tensor, norm_stats: dict) -> np.ndarray:
    """Convert normalised glucose values back to mg/dL using training statistics."""

    glucose_stats = _extract_glucose_stats(norm_stats)
    values_array = np.asarray(values, dtype=np.float32)
    return (values_array * float(glucose_stats["std"])) + float(glucose_stats["mean"])


def rmse(predictions, targets, norm_stats: dict | None = None) -> float:
    """Compute RMSE in mg/dL, denormalising first when statistics are provided."""

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)

    if norm_stats is not None:
        predictions_array = denormalise_glucose(predictions_array, norm_stats)
        targets_array = denormalise_glucose(targets_array, norm_stats)

    return float(np.sqrt(np.mean((predictions_array - targets_array) ** 2)))


def mae(predictions, targets, norm_stats: dict | None = None) -> float:
    """Compute MAE in mg/dL, denormalising first when statistics are provided."""

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)

    if norm_stats is not None:
        predictions_array = denormalise_glucose(predictions_array, norm_stats)
        targets_array = denormalise_glucose(targets_array, norm_stats)

    return float(np.mean(np.abs(predictions_array - targets_array)))


def _clarke_zone(actual: float, predicted: float) -> str:
    """Classify a single prediction into a Clarke Error Grid zone."""

    if (actual < 70 and predicted < 70) or abs(actual - predicted) < 0.2 * actual:
        return "A"
    if actual <= 70 and predicted >= 180:
        return "E"
    if actual >= 180 and predicted <= 70:
        return "E"
    if actual >= 240 and 70 <= predicted <= 180:
        return "D"
    if actual <= 70 <= predicted <= 180:
        return "D"
    if 70 <= actual <= 290 and predicted >= actual + 110:
        return "C"
    if 130 <= actual <= 180 and predicted <= ((7.0 / 5.0) * actual) - 182:
        return "C"
    if actual < predicted:
        return "B"
    return "B"


def _build_clarke_figure(predictions: np.ndarray, targets: np.ndarray, patient_id: Any) -> plt.Figure:
    """Create a Clarke-style scatter plot for qualitative inspection."""

    prediction_max = float(np.max(predictions)) if predictions.size else 0.0
    target_max = float(np.max(targets)) if targets.size else 0.0
    max_glucose = float(max(400.0, prediction_max, target_max) + 20.0)
    line_x = np.linspace(0.0, max_glucose, 400)

    figure, axis = plt.subplots(figsize=(8, 8))
    axis.scatter(targets, predictions, s=18, alpha=0.6, color="#1f77b4", edgecolor="none")

    axis.plot(line_x, line_x, color="black", linewidth=1.5, label="Ideal")
    axis.plot(line_x, 1.2 * line_x, color="green", linestyle="--", linewidth=1.0, label="20% bounds")
    axis.plot(line_x, 0.8 * line_x, color="green", linestyle="--", linewidth=1.0)
    axis.axhline(70.0, color="grey", linestyle=":", linewidth=1.0)
    axis.axvline(70.0, color="grey", linestyle=":", linewidth=1.0)
    axis.axhline(180.0, color="grey", linestyle=":", linewidth=1.0)
    axis.axvline(180.0, color="grey", linestyle=":", linewidth=1.0)
    axis.plot(line_x[(line_x >= 70.0) & (line_x <= 290.0)], line_x[(line_x >= 70.0) & (line_x <= 290.0)] + 110.0, color="#d62728", linestyle="-.", linewidth=1.0)
    lower_x = line_x[(line_x >= 130.0) & (line_x <= 180.0)]
    axis.plot(lower_x, ((7.0 / 5.0) * lower_x) - 182.0, color="#d62728", linestyle="-.", linewidth=1.0)

    axis.text(35, 30, "A", fontsize=16, fontweight="bold")
    axis.text(250, 120, "B", fontsize=16, fontweight="bold")
    axis.text(120, 310, "C", fontsize=16, fontweight="bold")
    axis.text(35, 150, "D", fontsize=16, fontweight="bold")
    axis.text(240, 35, "E", fontsize=16, fontweight="bold")

    axis.set_xlim(0.0, max_glucose)
    axis.set_ylim(0.0, max_glucose)
    axis.set_xlabel("Reference Glucose (mg/dL)")
    axis.set_ylabel("Predicted Glucose (mg/dL)")
    axis.set_title(f"Clarke Error Grid - Patient {patient_id}")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def clarke_error_grid(predictions, targets, patient_id) -> dict:
    """Compute Clarke Error Grid percentages and return a diagnostic figure."""

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)

    zone_counts = {zone: 0 for zone in ["A", "B", "C", "D", "E"]}
    for actual, predicted in zip(targets_array, predictions_array):
        zone_counts[_clarke_zone(float(actual), float(predicted))] += 1

    total_points = max(len(targets_array), 1)
    zone_percentages = {
        zone: (count / total_points) * 100.0 for zone, count in zone_counts.items()
    }
    zone_percentages["plot"] = _build_clarke_figure(predictions_array, targets_array, patient_id)
    return zone_percentages


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    norm_stats: dict,
    *,
    device: str,
    model_name: str = "Model",
) -> dict:
    """Run a full held-out evaluation and print a concise summary table."""

    model.eval()
    predictions_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    patient_horizon_predictions: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    patient_horizon_targets: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    cursor = 0
    with torch.no_grad():
        for hr_sequence, glucose_context, targets in test_loader:
            batch_size = hr_sequence.size(0)
            metadata = [
                test_loader.dataset.get_metadata(index)
                for index in range(cursor, cursor + batch_size)
            ]
            cursor += batch_size

            hr_sequence = hr_sequence.to(device)
            glucose_context = glucose_context.to(device)
            targets = targets.to(device)

            predictions = model(hr_sequence, glucose_context)

            predictions_np = predictions.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()

            predictions_batches.append(predictions_np)
            target_batches.append(targets_np)

            for batch_index, item_metadata in enumerate(metadata):
                patient_id = int(item_metadata["patient_id"])
                for horizon_index in range(targets_np.shape[1]):
                    patient_horizon_predictions[patient_id][horizon_index].append(
                        float(predictions_np[batch_index, horizon_index])
                    )
                    patient_horizon_targets[patient_id][horizon_index].append(
                        float(targets_np[batch_index, horizon_index])
                    )

    predictions_norm = np.concatenate(predictions_batches, axis=0)
    targets_norm = np.concatenate(target_batches, axis=0)
    predictions_mg_dl = denormalise_glucose(predictions_norm, norm_stats)
    targets_mg_dl = denormalise_glucose(targets_norm, norm_stats)

    metrics = {
        "model_name": model_name,
        "rmse_30min": rmse(predictions_norm[:, 0], targets_norm[:, 0], norm_stats),
        "rmse_60min": rmse(predictions_norm[:, 1], targets_norm[:, 1], norm_stats),
        "mae_30min": mae(predictions_norm[:, 0], targets_norm[:, 0], norm_stats),
        "mae_60min": mae(predictions_norm[:, 1], targets_norm[:, 1], norm_stats),
        "clarke_30min": clarke_error_grid(predictions_mg_dl[:, 0], targets_mg_dl[:, 0], "all"),
        "clarke_60min": clarke_error_grid(predictions_mg_dl[:, 1], targets_mg_dl[:, 1], "all"),
        "per_patient": {},
    }

    for patient_id in sorted(patient_horizon_predictions):
        patient_metrics = {}
        for horizon_index, horizon_label in enumerate(["30min", "60min"]):
            patient_predictions = np.asarray(patient_horizon_predictions[patient_id][horizon_index], dtype=np.float32)
            patient_targets = np.asarray(patient_horizon_targets[patient_id][horizon_index], dtype=np.float32)
            patient_predictions_mg_dl = denormalise_glucose(patient_predictions, norm_stats)
            patient_targets_mg_dl = denormalise_glucose(patient_targets, norm_stats)
            patient_metrics[horizon_label] = {
                "rmse": rmse(patient_predictions, patient_targets, norm_stats),
                "mae": mae(patient_predictions, patient_targets, norm_stats),
                "clarke": clarke_error_grid(
                    patient_predictions_mg_dl,
                    patient_targets_mg_dl,
                    patient_id,
                ),
            }
        metrics["per_patient"][str(patient_id)] = patient_metrics

    zone_ab_30 = metrics["clarke_30min"]["A"] + metrics["clarke_30min"]["B"]
    zone_ab_60 = metrics["clarke_60min"]["A"] + metrics["clarke_60min"]["B"]

    print()
    print(f"{model_name} Test Summary")
    print("-" * 56)
    print(f"{'Horizon':<12}{'RMSE':>12}{'MAE':>12}{'Zone A+B':>12}")
    print("-" * 56)
    print(f"{'30 min':<12}{metrics['rmse_30min']:>12.3f}{metrics['mae_30min']:>12.3f}{zone_ab_30:>12.2f}")
    print(f"{'60 min':<12}{metrics['rmse_60min']:>12.3f}{metrics['mae_60min']:>12.3f}{zone_ab_60:>12.2f}")
    print("-" * 56)
    print()

    return metrics


__all__ = [
    "clarke_error_grid",
    "denormalise_glucose",
    "evaluate_model",
    "mae",
    "rmse",
]
