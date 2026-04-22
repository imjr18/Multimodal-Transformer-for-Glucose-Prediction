"""Modality ablation experiments for Part B."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from part_b.evaluate import collect_predictions, summarise_predictions


def _condition_transform(condition: str):
    """Return the tensor transform corresponding to one ablation condition."""

    def transform(hr_seq, glucose_context, ecg_features, emg_features, targets):
        if condition == "no_ecg":
            ecg_features = torch.zeros_like(ecg_features)
        elif condition == "no_emg":
            emg_features = torch.zeros_like(emg_features)
        elif condition == "hr_only":
            ecg_features = torch.zeros_like(ecg_features)
            emg_features = torch.zeros_like(emg_features)
        elif condition == "no_hr_ctx":
            glucose_context = torch.zeros_like(glucose_context)
        return hr_seq, glucose_context, ecg_features, emg_features, targets

    return transform


def plot_modality_contribution(ablation_df: pd.DataFrame, save_path: str | Path) -> str:
    """Plot RMSE increases relative to the all-modalities baseline."""

    plot_df = ablation_df[ablation_df["condition"] != "all_modalities"].copy()

    figure, axis = plt.subplots(figsize=(9, 5))
    x_positions = range(len(plot_df))
    axis.bar(
        [position - 0.18 for position in x_positions],
        plot_df["rmse_increase_30min"],
        width=0.35,
        label="30 min",
        color="#4c78a8",
    )
    axis.bar(
        [position + 0.18 for position in x_positions],
        plot_df["rmse_increase_60min"],
        width=0.35,
        label="60 min",
        color="#f58518",
    )
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(plot_df["condition"], rotation=15, ha="right")
    axis.set_ylabel("RMSE Increase (mg/dL) vs All Modalities")
    axis.set_title("Modality Contribution via Ablation")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


def run_modality_ablation(
    model,
    test_loader,
    norm_stats: dict,
    *,
    device: str,
    csv_path: str | Path,
    figure_path: str | Path,
) -> pd.DataFrame:
    """Systematically remove modalities at test time and report the error change."""

    conditions = [
        "all_modalities",
        "no_ecg",
        "no_emg",
        "hr_only",
        "no_hr_ctx",
    ]

    rows: list[dict] = []
    baseline_metrics = None

    for condition in conditions:
        input_transform = None if condition == "all_modalities" else _condition_transform(condition)
        outputs = collect_predictions(
            model,
            test_loader,
            device=device,
            amp_enabled=(device == "cuda"),
            input_transform=input_transform,
            measure_inference=False,
        )
        metrics = summarise_predictions(
            outputs["predictions"],
            outputs["targets"],
            norm_stats,
            model_name=condition,
            verbose=False,
        )

        if baseline_metrics is None:
            baseline_metrics = metrics

        rows.append(
            {
                "condition": condition,
                "rmse_30min": metrics["rmse_30min"],
                "rmse_60min": metrics["rmse_60min"],
                "rmse_increase_30min": metrics["rmse_30min"] - baseline_metrics["rmse_30min"],
                "rmse_increase_60min": metrics["rmse_60min"] - baseline_metrics["rmse_60min"],
                "zone_AB_pct": metrics["zone_ab_pct"],
            }
        )

    ablation_df = pd.DataFrame(rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    ablation_df.to_csv(csv_path, index=False)

    plot_modality_contribution(ablation_df, figure_path)

    print()
    print("Modality Ablation Study")
    print("-" * 92)
    print(ablation_df.to_string(index=False, formatters={
        "rmse_30min": "{:.3f}".format,
        "rmse_60min": "{:.3f}".format,
        "rmse_increase_30min": "{:.3f}".format,
        "rmse_increase_60min": "{:.3f}".format,
        "zone_AB_pct": "{:.2f}".format,
    }))
    print("-" * 92)
    print()

    return ablation_df


__all__ = ["plot_modality_contribution", "run_modality_ablation"]
