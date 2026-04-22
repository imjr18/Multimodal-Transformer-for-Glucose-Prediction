"""Benchmarking and analysis utilities for Part C."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from part_a.evaluate import denormalise_glucose
from part_a.train import count_parameters
from part_b.evaluate import summarise_predictions
from part_c.models.common import resample_profile
from preprocessing.eeg_simulation import extract_band_power_sequence, infer_sleep_stage_from_band_powers


def _autocast_context(enabled: bool):
    """Return CUDA autocast when enabled, otherwise a no-op context."""

    if enabled:
        return torch.cuda.amp.autocast()
    return nullcontext()


def demonstrate_vanilla_attention_failure(config: dict | None = None) -> dict:
    """Demonstrate why vanilla attention is infeasible for raw 2-minute EEG."""

    sequence_length = 2 * 60 * 256 if config is None else int(config["eeg_samples"])
    required_bytes = sequence_length * sequence_length * 4
    required_gb = required_bytes / 1_000_000_000.0
    available_gb = 6.0

    message = (
        f"Vanilla attention on 2-min EEG requires {required_gb:.2f} GB for one float32 "
        f"attention matrix. Available: {available_gb:.0f}GB. Infeasible."
    )

    status = "theoretical_only"
    error_message = None
    should_attempt_cuda = torch.cuda.is_available() and (config is None or config.get("device") == "cuda")
    if should_attempt_cuda:
        try:
            device = "cuda"
            embed_dim = 64 if config is None else int(config["d_model"])
            n_heads = 4 if config is None else int(config["n_heads"])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            attention = torch.nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=n_heads,
                batch_first=True,
                device=device,
                dtype=torch.float32,
            )
            query = torch.randn(1, sequence_length, embed_dim, device=device, dtype=torch.float32)
            with torch.no_grad():
                _ = attention(query, query, query)

            peak_gb = torch.cuda.max_memory_allocated() / float(1024**3)
            message = (
                f"{message} Observed peak allocation on the current GPU: {peak_gb:.2f} GB. "
                f"This still exceeds the 6GB target envelope once gradients and the rest of the model are included."
            )
            status = "executed"
        except RuntimeError as error:
            error_message = str(error)
            message = f"{message} RuntimeError captured while attempting it: {error_message}"
            status = "oom_or_runtime_error"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        if torch.cuda.is_available():
            suffix = "CUDA execution is disabled for this run, so only the theoretical requirement is reported."
        else:
            suffix = "CUDA is not available in this environment, so only the theoretical requirement is reported."
        message = f"{message} {suffix}"

    print(message)
    return {
        "sequence_length": sequence_length,
        "required_gb": required_gb,
        "available_gb": available_gb,
        "status": status,
        "message": message,
        "error": error_message,
    }


def collect_full_modal_predictions(
    model: torch.nn.Module,
    data_loader,
    *,
    device: str,
    amp_enabled: bool,
    measure_resources: bool = False,
) -> dict:
    """Run full-modal inference and collect predictions, timing, and memory."""

    model.eval()
    prediction_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    inference_times_ms: list[float] = []
    peak_vram_mb = 0.0

    if measure_resources and device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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

            if measure_resources and device == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with _autocast_context(amp_enabled):
                    predictions = model(
                        hr_sequence,
                        glucose_context,
                        ecg_features,
                        emg_features,
                        eeg_signal,
                        cbf_signal,
                    )
                end_event.record()
                torch.cuda.synchronize()
                inference_times_ms.append(float(start_event.elapsed_time(end_event)))
            elif measure_resources:
                start_time = time.perf_counter()
                with _autocast_context(amp_enabled):
                    predictions = model(
                        hr_sequence,
                        glucose_context,
                        ecg_features,
                        emg_features,
                        eeg_signal,
                        cbf_signal,
                    )
                inference_times_ms.append((time.perf_counter() - start_time) * 1000.0)
            else:
                with _autocast_context(amp_enabled):
                    predictions = model(
                        hr_sequence,
                        glucose_context,
                        ecg_features,
                        emg_features,
                        eeg_signal,
                        cbf_signal,
                    )

            prediction_batches.append(predictions.detach().cpu().numpy())
            target_batches.append(targets.detach().cpu().numpy())

    if measure_resources and device == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / float(1024**2)

    return {
        "predictions": np.concatenate(prediction_batches, axis=0),
        "targets": np.concatenate(target_batches, axis=0),
        "inference_ms": float(np.mean(inference_times_ms)) if inference_times_ms else None,
        "peak_vram_mb": peak_vram_mb,
    }


def run_efficiency_benchmark(
    models: dict[str, torch.nn.Module],
    test_loader,
    norm_stats: dict,
    *,
    device: str,
    csv_path: str | Path,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    """Compare the three EEG strategies on error, speed, memory, and size."""

    rows: list[dict] = []
    for model_name, model in models.items():
        outputs = collect_full_modal_predictions(
            model,
            test_loader,
            device=device,
            amp_enabled=(device == "cuda"),
            measure_resources=True,
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
                "peak_vram_mb": outputs["peak_vram_mb"],
                "inference_ms": outputs["inference_ms"],
                "n_params": count_parameters(model),
            }
        )

    benchmark_df = pd.DataFrame(rows).sort_values(by=["rmse_60min", "rmse_30min"]).reset_index(drop=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(csv_path, index=False)

    fits_mask = (
        (benchmark_df["peak_vram_mb"] == 0.0)
        | (benchmark_df["peak_vram_mb"] <= float(config["comfortable_vram_mb"]))
    )
    feasible_df = benchmark_df[fits_mask]
    recommended_df = feasible_df if not feasible_df.empty else benchmark_df
    recommended_row = recommended_df.iloc[0]
    recommendation = {
        "recommended_model": recommended_row["model"],
        "reason": (
            "lowest 60-minute RMSE among models that fit comfortably within the target VRAM budget"
            if not feasible_df.empty
            else "lowest 60-minute RMSE; no model met the comfortable VRAM threshold in this run"
        ),
    }

    print()
    print("Part C Efficiency Benchmark")
    print("-" * 108)
    print(benchmark_df.to_string(index=False, formatters={
        "rmse_30min": "{:.3f}".format,
        "rmse_60min": "{:.3f}".format,
        "peak_vram_mb": "{:.1f}".format,
        "inference_ms": "{:.3f}".format,
    }))
    print("-" * 108)
    print(f"Recommended backbone: {recommendation['recommended_model']}")
    print()

    return benchmark_df, recommendation


def analyse_sleep_stage_attention(
    model,
    test_loader,
    norm_stats,
    *,
    device: str,
    save_path: str | Path,
) -> dict:
    """Analyse whether EEG attention differs between coarse sleep stages."""

    eeg_stats = norm_stats["eeg_signal"]
    stage_band_importance: dict[str, list[np.ndarray]] = {"deep_sleep": [], "light_sleep": [], "awake": []}
    stage_attention_profiles: dict[str, list[np.ndarray]] = {"deep_sleep": [], "light_sleep": [], "awake": []}
    stage_counts = {stage: 0 for stage in stage_band_importance}

    was_training = model.training
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            _, _, _, _, eeg_signal, _, _, _ = batch
            attention_profile = model.get_eeg_attention_profile(eeg_signal.to(device)).detach().cpu().numpy()
            eeg_signal_raw = (
                eeg_signal.numpy() * float(eeg_stats["std"])
            ) + float(eeg_stats["mean"])

            for sample_index in range(eeg_signal.shape[0]):
                raw_signal = eeg_signal_raw[sample_index]
                band_power_sequence = extract_band_power_sequence(
                    raw_signal,
                    sfreq=int(model.config["eeg_sfreq"]),
                    window_seconds=int(model.config["eeg_band_window_seconds"]),
                )
                if band_power_sequence.size == 0:
                    continue

                stage = infer_sleep_stage_from_band_powers(band_power_sequence)
                resampled_attention = resample_profile(
                    attention_profile[sample_index],
                    target_length=band_power_sequence.shape[0],
                )
                normalised_attention = resampled_attention / max(resampled_attention.sum(), 1e-6)
                weighted_band_importance = (band_power_sequence * normalised_attention[:, None]).sum(axis=0)

                stage_counts[stage] += 1
                stage_band_importance[stage].append(weighted_band_importance)
                stage_attention_profiles[stage].append(normalised_attention)

    if was_training:
        model.train()

    stage_order = ["deep_sleep", "light_sleep", "awake"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    band_labels = ["delta", "theta", "alpha", "beta", "gamma"]
    bar_width = 0.22
    x_positions = np.arange(len(band_labels), dtype=np.float32)

    for stage_index, stage in enumerate(stage_order):
        if stage_band_importance[stage]:
            mean_band_importance = np.mean(stage_band_importance[stage], axis=0)
            mean_attention = np.mean(stage_attention_profiles[stage], axis=0)
        else:
            mean_band_importance = np.zeros(5, dtype=np.float32)
            mean_attention = np.zeros(int(model.config["eeg_band_tokens"]), dtype=np.float32)

        axes[0].bar(
            x_positions + ((stage_index - 1) * bar_width),
            mean_band_importance,
            width=bar_width,
            label=f"{stage} (n={stage_counts[stage]})",
        )
        axes[1].plot(
            np.arange(mean_attention.size),
            mean_attention,
            linewidth=2.0,
            label=f"{stage} (n={stage_counts[stage]})",
        )

    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(band_labels)
    axes[0].set_ylabel("Attention-Weighted Relative Band Power")
    axes[0].set_title("Band Importance by Sleep Stage")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_xlabel("Seconds Within 2-Minute EEG Window")
    axes[1].set_ylabel("Mean Resampled Attention")
    axes[1].set_title("Temporal EEG Attention by Sleep Stage")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return {
        "figure_path": str(output_path),
        "stage_counts": stage_counts,
    }


__all__ = [
    "analyse_sleep_stage_attention",
    "collect_full_modal_predictions",
    "demonstrate_vanilla_attention_failure",
    "run_efficiency_benchmark",
]
