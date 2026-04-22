"""Integrated Gradients for the non-invasive multimodal model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from noninvasive_glucose.simulation.noninvasive_simulator import window_to_model_inputs


INPUT_KEYS = ["hr", "ecg_features", "emg_features", "eeg_bands", "cbf"]


def _prepare_baseline(sample: dict[str, torch.Tensor], baseline: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
    """Create zero baselines when none are supplied."""

    if baseline is not None:
        return {key: baseline[key].clone().detach().to(dtype=torch.float32) for key in INPUT_KEYS}
    return {key: torch.zeros_like(sample[key], dtype=torch.float32) for key in INPUT_KEYS}


def compute_integrated_gradients(
    model,
    sample: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor] | None = None,
    n_steps: int = 50,
) -> dict[str, Any]:
    """Compute IG attributions for one non-invasive window on CPU."""

    device = "cpu"
    was_training = model.training
    model = model.to(device)
    model.eval()

    batch = {key: value.detach().clone().to(device=device, dtype=torch.float32) for key, value in sample.items()}
    input_tensors = {key: batch[key] for key in INPUT_KEYS}
    baseline_tensors = _prepare_baseline(batch, baseline)
    total_grads = {key: torch.zeros_like(input_tensors[key], dtype=torch.float32) for key in INPUT_KEYS}

    def _forward(current_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _ = model(
            current_inputs["hr"],
            current_inputs["ecg_features"],
            current_inputs["emg_features"],
            current_inputs["eeg_bands"],
            current_inputs["cbf"],
            user_ids=batch["user_ids"],
            archetype_ids=batch["archetype_ids"],
        )
        return mean.mean()

    for step in range(1, n_steps + 1):
        alpha = float(step) / float(n_steps)
        scaled_inputs = {}
        for key in INPUT_KEYS:
            scaled = baseline_tensors[key] + alpha * (input_tensors[key] - baseline_tensors[key])
            scaled.requires_grad_(True)
            scaled_inputs[key] = scaled

        output = _forward(scaled_inputs)
        gradients = torch.autograd.grad(
            output,
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
        pred_input = _forward(input_tensors)
        pred_baseline = _forward(baseline_tensors)

    attribution_sum = float(sum(attributions[key].sum().item() for key in INPUT_KEYS))
    output_delta = float((pred_input - pred_baseline).item())
    completeness_error = attribution_sum - output_delta
    print(
        f"Non-invasive IG completeness: attribution_sum={attribution_sum:.6f}, "
        f"output_delta={output_delta:.6f}, error={completeness_error:.6f}"
    )

    if was_training:
        model.train()

    return {
        "hr_attributions": attributions["hr"].detach().cpu().numpy()[0].astype("float32"),
        "ecg_attributions": attributions["ecg_features"].detach().cpu().numpy()[0].astype("float32"),
        "emg_attributions": attributions["emg_features"].detach().cpu().numpy()[0].astype("float32"),
        "eeg_attributions": attributions["eeg_bands"].detach().cpu().numpy()[0].astype("float32"),
        "cbf_attributions": attributions["cbf"].detach().cpu().numpy()[0].astype("float32"),
        "completeness": {
            "attribution_sum": attribution_sum,
            "output_delta": output_delta,
            "error": completeness_error,
        },
    }


def _plot_attribution_heatmaps(
    scenario_name: str,
    mean_attributions: dict[str, np.ndarray],
    *,
    save_path: str | Path,
) -> str:
    """Plot one multi-panel attribution heatmap per physiological scenario."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    specs = [
        ("HR", mean_attributions["hr_attributions"]),
        ("ECG", mean_attributions["ecg_attributions"]),
        ("EMG", mean_attributions["emg_attributions"]),
        ("EEG", mean_attributions["eeg_attributions"]),
        ("CBF", mean_attributions["cbf_attributions"]),
    ]
    figure, axes = plt.subplots(1, len(specs), figsize=(18, 4.2))
    for axis, (label, matrix) in zip(axes, specs):
        image = axis.imshow(np.abs(matrix).T, aspect="auto", cmap="magma")
        axis.set_title(label)
        axis.set_xlabel("Time step")
        axis.set_ylabel("Feature")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Non-invasive Integrated Gradients - {scenario_name.replace('_', ' ').title()}")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def _scenario_windows(test_windows: list[dict], scenario_name: str, max_windows: int) -> list[dict]:
    """Select windows that match a named physiological scenario."""

    if scenario_name == "fasting_stable":
        candidates = [window for window in test_windows if bool(window.get("fasting_state"))]
    elif scenario_name == "post_meal":
        candidates = [window for window in test_windows if bool(window.get("post_meal_state"))]
    elif scenario_name == "post_exercise":
        candidates = [window for window in test_windows if bool(window.get("post_exercise_state"))]
    elif scenario_name == "deep_sleep":
        candidates = [window for window in test_windows if bool(window.get("deep_sleep_state"))]
    else:
        candidates = []
    return candidates[:max_windows]


def run_noninvasive_attribution(model, test_windows: list[dict], config: dict) -> dict:
    """Run state-conditioned IG attribution without glucose context."""

    scenario_names = ["fasting_stable", "post_meal", "post_exercise", "deep_sleep"]
    summary_rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}

    for scenario_name in scenario_names:
        scenario_windows = _scenario_windows(test_windows, scenario_name, max_windows=int(config["ig_windows_per_scenario"]))
        if not scenario_windows:
            results[scenario_name] = {"count": 0, "message": "No matching windows found."}
            continue

        ig_outputs = []
        for window in scenario_windows:
            sample = window_to_model_inputs(window, device="cpu")
            ig_outputs.append(
                compute_integrated_gradients(
                    model,
                    sample,
                    baseline=None,
                    n_steps=int(config["ig_steps"]),
                )
            )

        mean_attributions = {
            key: np.mean([output[key] for output in ig_outputs], axis=0).astype("float32")
            for key in ["hr_attributions", "ecg_attributions", "emg_attributions", "eeg_attributions", "cbf_attributions"]
        }
        modality_abs = {
            "hr_pct": float(np.sum(np.abs(mean_attributions["hr_attributions"]))),
            "ecg_pct": float(np.sum(np.abs(mean_attributions["ecg_attributions"]))),
            "emg_pct": float(np.sum(np.abs(mean_attributions["emg_attributions"]))),
            "eeg_pct": float(np.sum(np.abs(mean_attributions["eeg_attributions"]))),
            "cbf_pct": float(np.sum(np.abs(mean_attributions["cbf_attributions"]))),
        }
        total_abs = max(sum(modality_abs.values()), 1e-6)
        modality_pct = {key: (value / total_abs) * 100.0 for key, value in modality_abs.items()}
        mean_completeness_error = float(np.mean([output["completeness"]["error"] for output in ig_outputs]))

        figure_path = _plot_attribution_heatmaps(
            scenario_name,
            mean_attributions,
            save_path=Path(config["noninvasive_attr_dir"]) / f"{scenario_name}.png",
        )
        summary_rows.append(
            {
                "scenario": scenario_name,
                "n_windows": len(scenario_windows),
                "hr_pct": modality_pct["hr_pct"],
                "ecg_pct": modality_pct["ecg_pct"],
                "emg_pct": modality_pct["emg_pct"],
                "eeg_pct": modality_pct["eeg_pct"],
                "cbf_pct": modality_pct["cbf_pct"],
                "mean_completeness_error": mean_completeness_error,
                "plot_path": figure_path,
            }
        )
        results[scenario_name] = {
            "count": len(scenario_windows),
            "modality_percentages": modality_pct,
            "mean_completeness_error": mean_completeness_error,
            "plot_path": figure_path,
        }

    summary = pd.DataFrame(summary_rows)
    Path(config["noninvasive_attr_summary_path"]).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config["noninvasive_attr_summary_path"], index=False)
    return {"summary_table": summary, "results": results}


__all__ = ["compute_integrated_gradients", "run_noninvasive_attribution"]

