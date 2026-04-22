"""Evaluate calibration quality of the model's predictive uncertainty."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from noninvasive_glucose.evaluate.metrics import denormalise_glucose


def _acceptance_confidence(std_mg_dl: np.ndarray, band_mg_dl: float) -> np.ndarray:
    """Probability that a Gaussian prediction lies within +/- band of its mean."""

    std = np.clip(std_mg_dl, 1e-6, None)
    z = band_mg_dl / std
    return torch.erf(torch.tensor(z / math.sqrt(2.0), dtype=torch.float32)).numpy()


def evaluate_uncertainty_calibration(model, test_loader, norm_stats, config: dict) -> dict:
    """Measure coverage, sharpness, and reliability of predictive intervals."""

    device = config["device"]
    mean_predictions: list[np.ndarray] = []
    std_predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    stable_widths: list[float] = []
    exercise_widths: list[float] = []

    cursor = 0
    for batch in test_loader:
        hr = batch["hr"].to(device, non_blocking=True)
        ecg = batch["ecg_features"].to(device, non_blocking=True)
        emg = batch["emg_features"].to(device, non_blocking=True)
        eeg = batch["eeg_bands"].to(device, non_blocking=True)
        cbf = batch["cbf"].to(device, non_blocking=True)
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        archetype_ids = batch["archetype_ids"].to(device, non_blocking=True)
        target = batch["target"].detach().cpu().numpy()

        prediction_bundle = model.predict_with_uncertainty(
            hr,
            ecg,
            emg,
            eeg,
            cbf,
            user_ids=user_ids,
            archetype_ids=archetype_ids,
            n_samples=int(config["mc_dropout_samples"]),
        )

        mean_norm = prediction_bundle["mean"].detach().cpu().numpy()
        std_norm = prediction_bundle["total_std"].detach().cpu().numpy()
        mean_mg = denormalise_glucose(mean_norm, norm_stats)
        std_mg = std_norm * float(norm_stats["glucose_current"]["std"])

        mean_predictions.append(mean_mg)
        std_predictions.append(std_mg)
        targets.append(denormalise_glucose(target, norm_stats))

        for local_index in range(len(mean_mg)):
            metadata = test_loader.dataset.get_metadata(cursor + local_index)
            interval_width = float(3.92 * std_mg[local_index])
            if bool(metadata.get("fasting_state")):
                stable_widths.append(interval_width)
            if bool(metadata.get("post_exercise_state")):
                exercise_widths.append(interval_width)
        cursor += len(mean_mg)

    mean_array = np.concatenate(mean_predictions, axis=0)
    std_array = np.concatenate(std_predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)

    lower = mean_array - (1.96 * std_array)
    upper = mean_array + (1.96 * std_array)
    coverage = float(np.mean((target_array >= lower) & (target_array <= upper)) * 100.0)
    sharpness = float(np.mean(upper - lower))

    acceptance_band = float(config["uncertainty_acceptance_band"])
    predicted_confidence = _acceptance_confidence(std_array, acceptance_band)
    actual_accuracy = (np.abs(mean_array - target_array) <= acceptance_band).astype(np.float32)

    bins = np.linspace(0.0, 1.0, int(config["reliability_bins"]) + 1)
    bin_centers: list[float] = []
    bin_accuracies: list[float] = []
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (predicted_confidence >= left) & (predicted_confidence < right)
        if not np.any(mask):
            continue
        bin_centers.append(float((left + right) / 2.0))
        bin_accuracies.append(float(actual_accuracy[mask].mean()))

    figure_path = Path(config["reliability_diagram_path"])
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="grey", label="Ideal")
    axis.plot(bin_centers, bin_accuracies, marker="o", color="#1f77b4", label="Model")
    axis.set_xlabel("Predicted confidence of |error| <= 15 mg/dL")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Reliability Diagram")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    results = {
        "coverage_95_pct": coverage,
        "sharpness_mg_dl": sharpness,
        "mean_interval_width_fasting": float(np.mean(stable_widths)) if stable_widths else None,
        "mean_interval_width_post_exercise": float(np.mean(exercise_widths)) if exercise_widths else None,
        "reliability_bins": bin_centers,
        "reliability_accuracy": bin_accuracies,
        "reliability_diagram_path": str(figure_path),
    }

    Path(config["uncertainty_metrics_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(config["uncertainty_metrics_path"]).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


__all__ = ["evaluate_uncertainty_calibration"]
