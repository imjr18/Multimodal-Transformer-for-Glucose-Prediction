"""Compare the non-invasive model with simple reference baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from noninvasive_glucose.evaluate.metrics import clarke_error_grid, mae, rmse


def _extract_predictions(model, test_loader, *, use_uncertainty_mean: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Collect predictions and targets from the test loader."""

    device = getattr(model, "config", {}).get("device", "cpu")
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in test_loader:
        hr = batch["hr"].to(device, non_blocking=True)
        ecg = batch["ecg_features"].to(device, non_blocking=True)
        emg = batch["emg_features"].to(device, non_blocking=True)
        eeg = batch["eeg_bands"].to(device, non_blocking=True)
        cbf = batch["cbf"].to(device, non_blocking=True)
        user_ids = batch["user_ids"].to(device, non_blocking=True)
        archetype_ids = batch["archetype_ids"].to(device, non_blocking=True)
        target = batch["target"].detach().cpu().numpy()

        if use_uncertainty_mean:
            prediction_bundle = model.predict_with_uncertainty(
                hr,
                ecg,
                emg,
                eeg,
                cbf,
                user_ids=user_ids,
                archetype_ids=archetype_ids,
            )
            mean = prediction_bundle["mean"].detach().cpu().numpy()
        else:
            mean, _ = model(
                hr,
                ecg,
                emg,
                eeg,
                cbf,
                user_ids=user_ids,
                archetype_ids=archetype_ids,
            )
            mean = mean.detach().cpu().numpy()

        predictions.append(mean)
        targets.append(target)

    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def compare_against_baselines(model, test_loader, norm_stats: dict, config: dict) -> pd.DataFrame:
    """Compare the learned model with population mean and optional supervised reference."""

    model_predictions, targets = _extract_predictions(model, test_loader)
    rows = []

    ni_clarke = clarke_error_grid(
        predictions=np.asarray(model_predictions, dtype=np.float32) * float(norm_stats["glucose_current"]["std"]) + float(norm_stats["glucose_current"]["mean"]),
        targets=np.asarray(targets, dtype=np.float32) * float(norm_stats["glucose_current"]["std"]) + float(norm_stats["glucose_current"]["mean"]),
    )
    rows.append(
        {
            "model": "noninvasive_transformer",
            "rmse_mg_dl": rmse(model_predictions, targets, norm_stats),
            "mae_mg_dl": mae(model_predictions, targets, norm_stats),
            "zone_ab_pct": ni_clarke["A"] + ni_clarke["B"],
            "note": "Current-glucose estimate from biosignals only",
        }
    )

    population_mean_prediction = np.full_like(targets, fill_value=0.0, dtype=np.float32)
    pop_clarke = clarke_error_grid(
        predictions=np.asarray(population_mean_prediction, dtype=np.float32) * float(norm_stats["glucose_current"]["std"]) + float(norm_stats["glucose_current"]["mean"]),
        targets=np.asarray(targets, dtype=np.float32) * float(norm_stats["glucose_current"]["std"]) + float(norm_stats["glucose_current"]["mean"]),
    )
    rows.append(
        {
            "model": "population_mean",
            "rmse_mg_dl": rmse(population_mean_prediction, targets, norm_stats),
            "mae_mg_dl": mae(population_mean_prediction, targets, norm_stats),
            "zone_ab_pct": pop_clarke["A"] + pop_clarke["B"],
            "note": "Predict training-set mean glucose for every window",
        }
    )

    supervised_reference_path = Path(config["supervised_reference_path"])
    if supervised_reference_path.exists():
        supervised_payload = json.loads(supervised_reference_path.read_text(encoding="utf-8"))
        supervised_rmse = (
            supervised_payload.get("current_rmse")
            or supervised_payload.get("rmse")
            or supervised_payload.get("rmse_30min")
        )
        if supervised_rmse is not None:
            rows.append(
                {
                    "model": "supervised_reference",
                    "rmse_mg_dl": float(supervised_rmse),
                    "mae_mg_dl": supervised_payload.get("mae") or supervised_payload.get("mae_30min"),
                    "zone_ab_pct": supervised_payload.get("zone_ab_pct"),
                    "note": "External reference metric loaded from supervised project artifacts",
                }
            )

    comparison = pd.DataFrame(rows)
    Path(config["baseline_comparison_path"]).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(config["baseline_comparison_path"], index=False)
    return comparison


__all__ = ["compare_against_baselines"]
