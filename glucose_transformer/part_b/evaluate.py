"""Evaluation and comparison utilities for Part B."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from part_a.evaluate import denormalise_glucose, mae, rmse
from part_a.train import count_parameters


def _autocast_context(enabled: bool):
    """Return CUDA autocast when enabled, otherwise a no-op context."""

    if enabled:
        return torch.cuda.amp.autocast()
    return nullcontext()


def zone_ab_percentage(predictions_mg_dl: np.ndarray, targets_mg_dl: np.ndarray) -> float:
    """Compute Clarke Zone A+B percentage without constructing figures."""

    predictions = np.asarray(predictions_mg_dl, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets_mg_dl, dtype=np.float32).reshape(-1)

    zone_ab = 0
    for actual, predicted in zip(targets, predictions):
        in_zone_a = (actual < 70 and predicted < 70) or abs(actual - predicted) < 0.2 * actual
        if in_zone_a:
            zone_ab += 1
            continue

        in_zone_e = (actual <= 70 and predicted >= 180) or (actual >= 180 and predicted <= 70)
        in_zone_d = (actual >= 240 and 70 <= predicted <= 180) or (actual <= 70 <= predicted <= 180)
        in_zone_c = (70 <= actual <= 290 and predicted >= actual + 110) or (
            130 <= actual <= 180 and predicted <= ((7.0 / 5.0) * actual) - 182
        )
        if not (in_zone_e or in_zone_d or in_zone_c):
            zone_ab += 1

    return (zone_ab / max(len(targets), 1)) * 100.0


def collect_predictions(
    model: torch.nn.Module,
    data_loader,
    *,
    device: str,
    amp_enabled: bool,
    input_transform=None,
    measure_inference: bool = False,
) -> dict:
    """Run inference and return concatenated predictions, targets, and timing."""

    model.eval()
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    patient_id_batches: list[np.ndarray] = []
    timings_ms: list[float] = []

    with torch.no_grad():
        for hr_sequence, glucose_context, ecg_features, emg_features, targets, patient_ids in data_loader:
            hr_sequence = hr_sequence.to(device, non_blocking=True)
            glucose_context = glucose_context.to(device, non_blocking=True)
            ecg_features = ecg_features.to(device, non_blocking=True)
            emg_features = emg_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if input_transform is not None:
                hr_sequence, glucose_context, ecg_features, emg_features, targets = input_transform(
                    hr_sequence,
                    glucose_context,
                    ecg_features,
                    emg_features,
                    targets,
                )

            if measure_inference and device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with _autocast_context(amp_enabled):
                    predictions = model(hr_sequence, glucose_context, ecg_features, emg_features)
                end_event.record()
                torch.cuda.synchronize()
                timings_ms.append(float(start_event.elapsed_time(end_event)))
            elif measure_inference:
                start_time = time.perf_counter()
                with _autocast_context(amp_enabled):
                    predictions = model(hr_sequence, glucose_context, ecg_features, emg_features)
                timings_ms.append((time.perf_counter() - start_time) * 1000.0)
            else:
                with _autocast_context(amp_enabled):
                    predictions = model(hr_sequence, glucose_context, ecg_features, emg_features)

            prediction_batches.append(predictions.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())
            patient_id_batches.append(np.asarray(patient_ids, dtype=np.int64))

    predictions = np.concatenate(prediction_batches, axis=0)
    targets = np.concatenate(target_batches, axis=0)
    patient_ids = np.concatenate(patient_id_batches, axis=0)
    return {
        "predictions": predictions,
        "targets": targets,
        "patient_ids": patient_ids,
        "inference_ms": float(np.mean(timings_ms)) if timings_ms else None,
    }


def summarise_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    norm_stats: dict,
    *,
    model_name: str,
    verbose: bool = True,
) -> dict:
    """Compute denormalised Part B metrics from model outputs."""

    predictions_mg_dl = denormalise_glucose(predictions, norm_stats)
    targets_mg_dl = denormalise_glucose(targets, norm_stats)

    zone_ab_30 = zone_ab_percentage(predictions_mg_dl[:, 0], targets_mg_dl[:, 0])
    zone_ab_60 = zone_ab_percentage(predictions_mg_dl[:, 1], targets_mg_dl[:, 1])

    metrics = {
        "model_name": model_name,
        "rmse_30min": rmse(predictions[:, 0], targets[:, 0], norm_stats),
        "rmse_60min": rmse(predictions[:, 1], targets[:, 1], norm_stats),
        "mae_30min": mae(predictions[:, 0], targets[:, 0], norm_stats),
        "mae_60min": mae(predictions[:, 1], targets[:, 1], norm_stats),
        "zone_ab_30min": zone_ab_30,
        "zone_ab_60min": zone_ab_60,
        "zone_ab_pct": (zone_ab_30 + zone_ab_60) / 2.0,
    }

    if verbose:
        print()
        print(f"{model_name} Test Summary")
        print("-" * 68)
        print(f"{'Horizon':<12}{'RMSE':>12}{'MAE':>12}{'Zone A+B':>14}")
        print("-" * 68)
        print(f"{'30 min':<12}{metrics['rmse_30min']:>12.3f}{metrics['mae_30min']:>12.3f}{metrics['zone_ab_30min']:>14.2f}")
        print(f"{'60 min':<12}{metrics['rmse_60min']:>12.3f}{metrics['mae_60min']:>12.3f}{metrics['zone_ab_60min']:>14.2f}")
        print("-" * 68)
        print()

    return metrics


def evaluate_multimodal_model(
    model: torch.nn.Module,
    test_loader,
    norm_stats: dict,
    *,
    device: str,
    model_name: str = "Model",
    input_transform=None,
    verbose: bool = True,
) -> dict:
    """Run full multimodal evaluation on the test set."""

    outputs = collect_predictions(
        model,
        test_loader,
        device=device,
        amp_enabled=(device == "cuda"),
        input_transform=input_transform,
        measure_inference=False,
    )
    return summarise_predictions(
        outputs["predictions"],
        outputs["targets"],
        norm_stats,
        model_name=model_name,
        verbose=verbose,
    )


def compare_fusion_strategies(
    early_model: torch.nn.Module,
    late_model: torch.nn.Module,
    cross_model: torch.nn.Module,
    test_loader,
    norm_stats: dict,
    *,
    device: str,
    csv_path: str | Path,
) -> pd.DataFrame:
    """Compare early, late, and cross-attention fusion side by side."""

    rows: list[dict] = []
    for model_name, model in [
        ("EarlyFusionTransformer", early_model),
        ("LateFusionTransformer", late_model),
        ("CrossModalTransformer", cross_model),
    ]:
        outputs = collect_predictions(
            model,
            test_loader,
            device=device,
            amp_enabled=(device == "cuda"),
            measure_inference=True,
        )
        metrics = summarise_predictions(
            outputs["predictions"],
            outputs["targets"],
            norm_stats,
            model_name=model_name,
            verbose=False,
        )
        rows.append(
            {
                "model": model_name,
                "rmse_30min": metrics["rmse_30min"],
                "rmse_60min": metrics["rmse_60min"],
                "zone_AB_pct": metrics["zone_ab_pct"],
                "params": count_parameters(model),
                "inference_ms": outputs["inference_ms"],
            }
        )

    comparison = pd.DataFrame(rows).sort_values(by="rmse_60min", ascending=True).reset_index(drop=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)

    print()
    print("Fusion Strategy Comparison")
    print("-" * 92)
    print(comparison.to_string(index=False, formatters={
        "rmse_30min": "{:.3f}".format,
        "rmse_60min": "{:.3f}".format,
        "zone_AB_pct": "{:.2f}".format,
        "inference_ms": "{:.3f}".format,
    }))
    print("-" * 92)
    print()

    return comparison


def save_cross_attention_heatmap(
    model,
    dataset,
    *,
    device: str,
    save_path: str | Path,
    sample_index: int = 0,
) -> str:
    """Save a 24x24 HR-to-ECG cross-attention heatmap for one test sample."""

    hr_sequence, glucose_context, ecg_features, emg_features, _, _ = dataset[sample_index]
    attention_weights = model.get_cross_attention_weights(
        hr_sequence.unsqueeze(0).to(device),
        glucose_context.unsqueeze(0).to(device),
        ecg_features.unsqueeze(0).to(device),
        emg_features.unsqueeze(0).to(device),
    )
    attention_matrix = attention_weights["hr_to_ecg"][0].mean(dim=0).numpy()

    figure, axis = plt.subplots(figsize=(7.5, 6.5))
    image = axis.imshow(attention_matrix, cmap=plt.cm.RdBu_r, aspect="auto")
    axis.set_title("HR to ECG Cross-Attention")
    axis.set_xlabel("ECG Key Positions")
    axis.set_ylabel("HR Query Positions")
    axis.set_xticks(range(attention_matrix.shape[1]))
    axis.set_yticks(range(attention_matrix.shape[0]))
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


__all__ = [
    "collect_predictions",
    "compare_fusion_strategies",
    "evaluate_multimodal_model",
    "save_cross_attention_heatmap",
    "summarise_predictions",
    "zone_ab_percentage",
]
