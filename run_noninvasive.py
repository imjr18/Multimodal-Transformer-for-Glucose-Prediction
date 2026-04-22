"""Entry point for the standalone non-invasive glucose estimation pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from noninvasive_glucose.calibration.calibrate import UserCalibrator
from noninvasive_glucose.config import CONFIG, ensure_directories
from noninvasive_glucose.evaluate.compare_baselines import compare_against_baselines
from noninvasive_glucose.evaluate.metrics import calibration_improvement, clarke_error_grid, mae, rmse
from noninvasive_glucose.evaluate.uncertainty_eval import evaluate_uncertainty_calibration
from noninvasive_glucose.interpretability.noninvasive_ig import run_noninvasive_attribution
from noninvasive_glucose.models.noninvasive_transformer import NonInvasiveTransformer
from noninvasive_glucose.simulation.calibration_simulator import generate_calibration_session
from noninvasive_glucose.simulation.noninvasive_simulator import (
    apply_normalisation,
    build_processed_datasets,
    denormalise_glucose,
    load_processed_datasets,
    load_synthetic_cohort,
    window_to_model_inputs,
)
from noninvasive_glucose.training.finetune import (
    create_data_loader,
    load_pretrained_weights,
    load_trained_model,
    train_noninvasive_model,
)
from noninvasive_glucose.training.pretrain_ecg import pretrain_ecg_encoder
from noninvasive_glucose.training.pretrain_eeg import pretrain_eeg_encoder


def _device_overrides(config: dict, *, no_cuda: bool) -> dict:
    """Apply command-line device overrides to the configuration."""

    updated = dict(config)
    if no_cuda:
        updated["device"] = "cpu"
    return updated


def _evaluate_current_glucose(model, loader, norm_stats: dict) -> dict:
    """Run current-glucose evaluation on one loader."""

    device = model.config["device"]
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
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
                n_samples=int(model.config["mc_dropout_samples"]),
            )
            predictions.append(prediction_bundle["mean"].detach().cpu().numpy())
            targets.append(target)

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    predictions_mg = denormalise_glucose(prediction_array, norm_stats)
    targets_mg = denormalise_glucose(target_array, norm_stats)
    clarke = clarke_error_grid(predictions_mg, targets_mg)
    return {
        "rmse_mg_dl": rmse(prediction_array, target_array, norm_stats),
        "mae_mg_dl": mae(prediction_array, target_array, norm_stats),
        "zone_A_pct": clarke["A"],
        "zone_B_pct": clarke["B"],
        "zone_AB_pct": clarke["A"] + clarke["B"],
    }


def _evaluate_windows(model, windows: list[dict], norm_stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a list of processed windows and return predictions and targets."""

    device = model.config["device"]
    predictions: list[float] = []
    targets: list[float] = []
    for window in windows:
        inputs = window_to_model_inputs(window, device=device)
        bundle = model.predict_with_uncertainty(
            inputs["hr"],
            inputs["ecg_features"],
            inputs["emg_features"],
            inputs["eeg_bands"],
            inputs["cbf"],
            user_ids=inputs["user_ids"],
            archetype_ids=inputs["archetype_ids"],
        )
        predictions.append(float(bundle["mean"].detach().cpu().item()))
        targets.append(float(window["glucose_current"]))
    return np.asarray(predictions, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def _run_calibration_demo(model, processed_test_windows: list[dict], norm_stats: dict, config: dict) -> dict:
    """Simulate new-user calibration on one held-out synthetic user."""

    user_id = int(processed_test_windows[0]["user_id"])
    cohort = load_synthetic_cohort(config)
    user_record = next(record for record in cohort if int(record["user_id"]) == user_id)
    calibration_raw_pairs = generate_calibration_session(user_record, n_readings=int(config["n_calibration_readings"]))
    calibration_pairs = []
    for raw_window, glucose_value in calibration_raw_pairs:
        processed_window = apply_normalisation([raw_window], norm_stats)[0]
        calibration_pairs.append((processed_window, glucose_value))

    user_windows = [window for window in processed_test_windows if int(window["user_id"]) == user_id]
    before_predictions, targets = _evaluate_windows(model, user_windows, norm_stats)
    calibrator = UserCalibrator(model, config, norm_stats=norm_stats)
    calibrated_model = calibrator.calibrate(calibration_pairs)
    after_predictions, _ = _evaluate_windows(calibrated_model, user_windows, norm_stats)
    return calibration_improvement(before_predictions, after_predictions, targets, norm_stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone non-invasive glucose estimation pipeline")
    parser.add_argument("--simulate_only", action="store_true", help="Only build the synthetic cohort and processed windows.")
    parser.add_argument("--pretrain_only", action="store_true", help="Run encoder pretraining only.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only evaluate a saved checkpoint.")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU execution.")
    parser.add_argument("--force_rebuild", action="store_true", help="Rebuild synthetic data and overwrite processed caches.")
    args = parser.parse_args()

    config = _device_overrides(CONFIG, no_cuda=args.no_cuda)
    ensure_directories(config)

    datasets = build_processed_datasets(config, force=args.force_rebuild) if args.force_rebuild else load_processed_datasets(config) if Path(config["train_windows_path"]).exists() else build_processed_datasets(config)
    train_windows = datasets["train_windows"]
    val_windows = datasets["val_windows"]
    test_windows = datasets["test_windows"]
    norm_stats = datasets["norm_stats"]

    print(f"Processed windows | train={len(train_windows)} val={len(val_windows)} test={len(test_windows)}")
    if args.simulate_only:
        return

    model = NonInvasiveTransformer(config, n_users=sum(int(count) for count in config["archetype_counts"].values()))
    train_user_ids = sorted({int(window["user_id"]) for window in train_windows})
    model.user_embeddings.set_known_user_ids(train_user_ids)

    if not args.eval_only:
        print("Pretraining EEG encoder...")
        pretrain_eeg_encoder(
            model.eeg_encoder,
            train_windows,
            n_epochs=30,
            config=config,
            save_path=config["eeg_pretrain_checkpoint"],
        )
        print("Pretraining ECG encoder...")
        pretrain_ecg_encoder(
            model.ecg_encoder,
            train_windows,
            n_epochs=20,
            config=config,
            save_path=config["ecg_pretrain_checkpoint"],
        )
        if args.pretrain_only:
            return

    load_pretrained_weights(model, config)
    model.norm_stats = norm_stats

    if not args.eval_only:
        print("Fine-tuning full non-invasive model...")
        train_summary = train_noninvasive_model(model, train_windows, val_windows, norm_stats, config)
        print(f"Best validation RMSE: {train_summary['best_val_rmse']:.3f} mg/dL at epoch {train_summary['best_epoch']}")

    checkpoint = load_trained_model(model, config["model_checkpoint"], device=config["device"])
    model.config = checkpoint.get("config", config)
    model.norm_stats = checkpoint.get("norm_stats", norm_stats)
    model.user_embeddings.set_known_user_ids(train_user_ids)

    test_loader = create_data_loader(test_windows, config, shuffle=False)
    test_metrics = _evaluate_current_glucose(model, test_loader, norm_stats)
    uncertainty_metrics = evaluate_uncertainty_calibration(model, test_loader, norm_stats, config)
    baseline_table = compare_against_baselines(model, test_loader, norm_stats, config)
    calibration_metrics = _run_calibration_demo(model, test_windows, norm_stats, config)
    attribution_results = run_noninvasive_attribution(model, test_windows, config)

    summary_payload = {
        "test_metrics": test_metrics,
        "uncertainty_metrics": uncertainty_metrics,
        "calibration_metrics": calibration_metrics,
        "attribution_summary_path": config["noninvasive_attr_summary_path"],
        "baseline_comparison_path": config["baseline_comparison_path"],
    }
    summary_path = Path(config["results_dir"]) / "run_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print()
    print("Non-invasive Test Summary")
    print("-" * 56)
    print(f"RMSE:       {test_metrics['rmse_mg_dl']:.3f} mg/dL")
    print(f"MAE:        {test_metrics['mae_mg_dl']:.3f} mg/dL")
    print(f"Zone A+B:   {test_metrics['zone_AB_pct']:.2f}%")
    print(f"Coverage95: {uncertainty_metrics['coverage_95_pct']:.2f}%")
    print(f"Sharpness:  {uncertainty_metrics['sharpness_mg_dl']:.3f} mg/dL")
    print(f"Calib gain: {calibration_metrics['rmse_improvement']:.3f} mg/dL")
    print("-" * 56)
    print(baseline_table.to_string(index=False))
    if isinstance(attribution_results["summary_table"], pd.DataFrame) and not attribution_results["summary_table"].empty:
        print()
        print(attribution_results["summary_table"].to_string(index=False))


if __name__ == "__main__":
    main()
