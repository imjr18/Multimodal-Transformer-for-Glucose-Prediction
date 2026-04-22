"""Integrated-gradients attribution analysis for Part E."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from part_e.biological_validation import find_biological_scenario_windows
from part_e.common import make_single_window_batch, normalise_batch, save_json


INPUT_KEYS = [
    "hr_sequence",
    "glucose_context",
    "ecg_features",
    "emg_features",
    "eeg_signal",
    "cbf_signal",
]


def _prepare_baseline(sample: dict[str, Any], baseline: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    """Create baseline tensors for integrated gradients."""

    if baseline is not None:
        return {key: baseline[key].clone().detach().to(dtype=torch.float32) for key in INPUT_KEYS}
    return {key: torch.zeros_like(sample[key], dtype=torch.float32) for key in INPUT_KEYS}


def _aggregate_eeg_attributions(model, sample: dict[str, Any], raw_eeg_attr: torch.Tensor) -> np.ndarray:
    """Map raw EEG attributions to the encoder's native analysis granularity."""

    eeg_attr = raw_eeg_attr.detach().cpu().float()
    eeg_signal = sample["eeg_signal"].detach().cpu().float()
    eeg_encoder = model.backbone.eeg_encoder

    if hasattr(eeg_encoder, "_band_power_tokens"):
        band_tokens = eeg_encoder._band_power_tokens(eeg_signal).detach().cpu().numpy()[0]
        per_second_signed = eeg_attr.view(1, eeg_encoder.n_tokens, eeg_encoder.samples_per_window).sum(dim=-1).numpy()[0]
        return (per_second_signed[:, None] * band_tokens).astype("float32")

    if hasattr(eeg_encoder, "_patchify"):
        patches = eeg_attr.view(1, eeg_encoder.n_patches, eeg_encoder.patch_size).sum(dim=-1, keepdim=True)
        return patches.numpy()[0].astype("float32")

    if hasattr(eeg_encoder, "_local_summaries"):
        windows = eeg_attr.view(1, eeg_encoder.local_windows, eeg_encoder.local_window_samples).sum(dim=-1, keepdim=True)
        return windows.numpy()[0].astype("float32")

    raise ValueError(f"Unsupported EEG encoder for attribution aggregation: {type(eeg_encoder).__name__}")


def compute_integrated_gradients(model, sample, baseline=None, n_steps: int = 50) -> dict:
    """Compute integrated-gradients attributions for one Part D window."""

    device = "cpu"
    was_training = model.training
    model = model.to(device)
    model.eval()
    batch = normalise_batch(sample, device=device)

    input_tensors = {
        key: batch[key].detach().clone().to(dtype=torch.float32)
        for key in INPUT_KEYS
    }
    baseline_tensors = _prepare_baseline(batch, baseline)
    total_grads = {key: torch.zeros_like(tensor, dtype=torch.float32) for key, tensor in input_tensors.items()}

    def _forward_from_inputs(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        predictions = model(
            current_inputs["hr_sequence"],
            current_inputs["glucose_context"],
            current_inputs["ecg_features"],
            current_inputs["emg_features"],
            current_inputs["eeg_signal"],
            current_inputs["cbf_signal"],
            user_ids=batch["user_ids"],
            archetype_ids=batch["archetype_ids"],
        )
        return predictions.mean()

    for step in range(1, n_steps + 1):
        alpha = float(step) / float(n_steps)
        scaled_inputs = {}
        for key in INPUT_KEYS:
            scaled = baseline_tensors[key] + alpha * (input_tensors[key] - baseline_tensors[key])
            scaled.requires_grad_(True)
            scaled_inputs[key] = scaled

        prediction_scalar = _forward_from_inputs(scaled_inputs)
        gradients = torch.autograd.grad(
            prediction_scalar,
            tuple(scaled_inputs[key] for key in INPUT_KEYS),
            retain_graph=False,
            create_graph=False,
        )

        for key, gradient in zip(INPUT_KEYS, gradients):
            total_grads[key] += gradient.detach()

    average_grads = {key: total_grads[key] / float(n_steps) for key in INPUT_KEYS}
    attributions = {
        key: (input_tensors[key] - baseline_tensors[key]) * average_grads[key]
        for key in INPUT_KEYS
    }

    with torch.no_grad():
        prediction_input = _forward_from_inputs(input_tensors)
        prediction_baseline = _forward_from_inputs(baseline_tensors)

    total_attribution = sum(float(attributions[key].sum().item()) for key in INPUT_KEYS)
    delta_output = float((prediction_input - prediction_baseline).item())
    completeness_error = total_attribution - delta_output
    print(
        f"Integrated Gradients completeness: attribution_sum={total_attribution:.6f}, "
        f"output_delta={delta_output:.6f}, error={completeness_error:.6f}"
    )

    result = {
        "hr_attributions": torch.cat(
            [attributions["hr_sequence"], attributions["glucose_context"]],
            dim=-1,
        ).detach().cpu().numpy()[0].astype("float32"),
        "ecg_attributions": attributions["ecg_features"].detach().cpu().numpy()[0].astype("float32"),
        "emg_attributions": attributions["emg_features"].detach().cpu().numpy()[0].astype("float32"),
        "eeg_attributions": _aggregate_eeg_attributions(model, batch, attributions["eeg_signal"]),
        "cbf_attributions": attributions["cbf_signal"].detach().cpu().numpy()[0].astype("float32"),
        "completeness": {
            "attribution_sum": total_attribution,
            "output_delta": delta_output,
            "error": completeness_error,
        },
    }

    if was_training:
        model.train()
    return result


def _plot_scenario_heatmaps(
    scenario_name: str,
    mean_attributions: dict[str, np.ndarray],
    *,
    save_path: str | Path,
) -> str:
    """Plot one multi-panel heatmap summarising a biological scenario."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    modality_specs = [
        ("HR+Glucose", mean_attributions["hr_attributions"]),
        ("ECG", mean_attributions["ecg_attributions"]),
        ("EMG", mean_attributions["emg_attributions"]),
        ("EEG", mean_attributions["eeg_attributions"]),
        ("CBF", mean_attributions["cbf_attributions"]),
    ]

    figure, axes = plt.subplots(1, len(modality_specs), figsize=(18, 4.2))
    for axis, (label, matrix) in zip(axes, modality_specs):
        image = axis.imshow(np.abs(matrix).T, aspect="auto", cmap="magma")
        axis.set_title(label)
        axis.set_xlabel("Time / Token")
        axis.set_ylabel("Feature")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle(f"Integrated Gradients - {scenario_name.replace('_', ' ').title()}")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def run_biological_scenarios(model, test_data, norm_stats, *, config: dict) -> dict:
    """Run IG attribution analysis on the four Part E biological scenarios."""

    scenarios = find_biological_scenario_windows(
        test_data,
        split="test",
        max_matches=int(config["scenario_matches_per_condition"]),
        max_windows_per_user=int(config["analysis_max_windows_per_user"]),
    )

    scenario_rows: list[dict[str, Any]] = []
    serialisable_results: dict[str, Any] = {}

    for scenario_name, window_entries in scenarios.items():
        if not window_entries:
            serialisable_results[scenario_name] = {"count": 0, "message": "No matching windows found."}
            continue

        ig_results = []
        for window_entry in window_entries:
            batch = make_single_window_batch(window_entry, device="cpu")
            ig_results.append(
                compute_integrated_gradients(
                    model,
                    batch,
                    baseline=None,
                    n_steps=int(config["ig_n_steps"]),
                )
            )

        mean_attributions = {
            key: np.mean([result[key] for result in ig_results], axis=0).astype("float32")
            for key in ["hr_attributions", "ecg_attributions", "emg_attributions", "eeg_attributions", "cbf_attributions"]
        }

        modality_abs_totals = {
            "hr_pct": float(np.sum(np.abs(mean_attributions["hr_attributions"]))),
            "ecg_pct": float(np.sum(np.abs(mean_attributions["ecg_attributions"]))),
            "emg_pct": float(np.sum(np.abs(mean_attributions["emg_attributions"]))),
            "eeg_pct": float(np.sum(np.abs(mean_attributions["eeg_attributions"]))),
            "cbf_pct": float(np.sum(np.abs(mean_attributions["cbf_attributions"]))),
        }
        total_abs = max(sum(modality_abs_totals.values()), 1e-6)
        modality_percentages = {
            key: (value / total_abs) * 100.0
            for key, value in modality_abs_totals.items()
        }
        completeness_error = float(np.mean([result["completeness"]["error"] for result in ig_results]))

        figure_path = _plot_scenario_heatmaps(
            scenario_name,
            mean_attributions,
            save_path=Path(config["ig_scenarios_dir"]) / f"{scenario_name}.png",
        )

        scenario_rows.append(
            {
                "scenario": scenario_name,
                "n_windows": len(window_entries),
                "hr_pct": modality_percentages["hr_pct"],
                "ecg_pct": modality_percentages["ecg_pct"],
                "emg_pct": modality_percentages["emg_pct"],
                "eeg_pct": modality_percentages["eeg_pct"],
                "cbf_pct": modality_percentages["cbf_pct"],
                "mean_completeness_error": completeness_error,
                "plot_path": figure_path,
            }
        )
        serialisable_results[scenario_name] = {
            "count": len(window_entries),
            "mean_completeness_error": completeness_error,
            "modality_percentages": modality_percentages,
            "plot_path": figure_path,
        }

    summary_df = pd.DataFrame(scenario_rows)
    Path(config["ig_scenario_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(config["ig_scenario_summary_path"], index=False)
    save_json(serialisable_results, config["ig_scenario_results_path"])

    return {
        "summary_table": summary_df,
        "results": serialisable_results,
    }


__all__ = ["compute_integrated_gradients", "run_biological_scenarios"]
