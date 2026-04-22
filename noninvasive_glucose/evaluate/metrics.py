"""Metrics for non-invasive glucose estimation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def denormalise_glucose(values: np.ndarray | float, norm_stats: dict) -> np.ndarray:
    """Convert normalised glucose predictions back to mg/dL."""

    stats = norm_stats["glucose_current"]
    values_array = np.asarray(values, dtype=np.float32)
    return (values_array * float(stats["std"])) + float(stats["mean"])


def rmse(predictions, targets, norm_stats: dict | None = None) -> float:
    """Compute RMSE, denormalising first when statistics are supplied."""

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    if norm_stats is not None:
        predictions_array = denormalise_glucose(predictions_array, norm_stats)
        targets_array = denormalise_glucose(targets_array, norm_stats)
    return float(np.sqrt(np.mean((predictions_array - targets_array) ** 2)))


def mae(predictions, targets, norm_stats: dict | None = None) -> float:
    """Compute MAE, denormalising first when statistics are supplied."""

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    if norm_stats is not None:
        predictions_array = denormalise_glucose(predictions_array, norm_stats)
        targets_array = denormalise_glucose(targets_array, norm_stats)
    return float(np.mean(np.abs(predictions_array - targets_array)))


def _clarke_zone(actual: float, predicted: float) -> str:
    """Classify one point in the Clarke Error Grid."""

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
    return "B"


def _build_clarke_figure(predictions: np.ndarray, targets: np.ndarray) -> plt.Figure:
    """Create a Clarke-style scatter plot for current-glucose estimation."""

    max_glucose = float(max(400.0, predictions.max(initial=0.0), targets.max(initial=0.0)) + 20.0)
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
    axis.set_xlim(0.0, max_glucose)
    axis.set_ylim(0.0, max_glucose)
    axis.set_xlabel("Reference Glucose (mg/dL)")
    axis.set_ylabel("Predicted Glucose (mg/dL)")
    axis.set_title("Clarke Error Grid - Non-invasive Current Glucose Estimation")
    axis.legend(loc="upper left")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def clarke_error_grid(predictions, targets) -> dict[str, Any]:
    """Compute Clarke zone percentages and a diagnostic figure."""

    predictions_array = np.asarray(predictions, dtype=np.float32)
    targets_array = np.asarray(targets, dtype=np.float32)
    zone_counts = {zone: 0 for zone in ["A", "B", "C", "D", "E"]}
    for actual, predicted in zip(targets_array, predictions_array):
        zone_counts[_clarke_zone(float(actual), float(predicted))] += 1
    total_points = max(len(targets_array), 1)
    zone_percentages = {zone: (count / total_points) * 100.0 for zone, count in zone_counts.items()}
    zone_percentages["plot"] = _build_clarke_figure(predictions_array, targets_array)
    return zone_percentages


def calibration_improvement(before_predictions, after_predictions, targets, norm_stats: dict) -> dict[str, float]:
    """Summarise the RMSE change achieved by post-training calibration."""

    before_rmse = rmse(before_predictions, targets, norm_stats)
    after_rmse = rmse(after_predictions, targets, norm_stats)
    improvement = before_rmse - after_rmse
    relative_improvement = (improvement / before_rmse) * 100.0 if before_rmse > 0 else 0.0
    return {
        "before_rmse": before_rmse,
        "after_rmse": after_rmse,
        "rmse_improvement": improvement,
        "relative_improvement_pct": relative_improvement,
    }


__all__ = ["calibration_improvement", "clarke_error_grid", "denormalise_glucose", "mae", "rmse"]

