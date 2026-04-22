"""Evaluation utilities for Part D personalisation experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part_d.maml import evaluate_task_after_adaptation


def _mean_rmse(metrics: dict) -> float:
    """Return the mean of the two glucose-forecasting horizons."""

    return float((metrics["rmse_30min"] + metrics["rmse_60min"]) / 2.0)


def plot_adaptation_curve(
    model,
    test_tasks,
    norm_stats,
    *,
    config: dict,
    device: str,
    max_support_minutes: int = 60,
    save_path: str | Path,
) -> plt.Figure:
    """Plot RMSE as a function of available support-set duration."""

    support_windows = [size for size in [0, 1, 3, 6, 12] if (size * 5) <= int(max_support_minutes)]
    rows: list[dict] = []

    for task in test_tasks:
        for support_size in support_windows:
            metrics = evaluate_task_after_adaptation(
                model,
                task,
                norm_stats,
                config,
                device=device,
                support_size=support_size,
                use_user_embedding_override=True,
            )
            rows.append(
                {
                    "user_id": metrics["user_id"],
                    "archetype": metrics["archetype"],
                    "support_windows": support_size,
                    "support_minutes": support_size * 5,
                    "rmse_mean": _mean_rmse(metrics),
                }
            )

    results = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for archetype, group in results.groupby("archetype", sort=True):
        summary = (
            group.groupby("support_minutes")["rmse_mean"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("support_minutes")
        )
        x = summary["support_minutes"].to_numpy(dtype=np.float32)
        y = summary["mean"].to_numpy(dtype=np.float32)
        y_std = summary["std"].fillna(0.0).to_numpy(dtype=np.float32)
        axis.plot(x, y, marker="o", linewidth=2.0, label=archetype)
        axis.fill_between(x, y - y_std, y + y_std, alpha=0.18)

    axis.axhline(20.0, color="#c0392b", linestyle="--", linewidth=1.4, label="20 mg/dL threshold")
    axis.set_xlabel("Support Set Duration (minutes)")
    axis.set_ylabel("Mean RMSE Across 30- and 60-Min Horizons (mg/dL)")
    axis.set_title("Part D Adaptation Curve")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    return figure


def cross_archetype_evaluation(
    model,
    test_tasks,
    norm_stats,
    *,
    config: dict,
    device: str,
    csv_path: str | Path,
) -> pd.DataFrame:
    """Break down adaptation quality by archetype."""

    rows: list[dict] = []
    support_sizes = {"rmse_0shot": 0, "rmse_30min": 6, "rmse_60min": 12}

    archetypes = sorted({task["archetype"] for task in test_tasks})
    for archetype in archetypes:
        archetype_tasks = [task for task in test_tasks if task["archetype"] == archetype]
        aggregated_metrics: dict[str, list[float]] = {column: [] for column in support_sizes}

        for label, support_size in support_sizes.items():
            for task in archetype_tasks:
                metrics = evaluate_task_after_adaptation(
                    model,
                    task,
                    norm_stats,
                    config,
                    device=device,
                    support_size=support_size,
                    use_user_embedding_override=True,
                )
                aggregated_metrics[label].append(_mean_rmse(metrics))

        rmse_0shot = float(np.mean(aggregated_metrics["rmse_0shot"]))
        rmse_30min = float(np.mean(aggregated_metrics["rmse_30min"]))
        rmse_60min = float(np.mean(aggregated_metrics["rmse_60min"]))
        rows.append(
            {
                "archetype": archetype,
                "rmse_0shot": rmse_0shot,
                "rmse_30min": rmse_30min,
                "rmse_60min": rmse_60min,
                "rmse_improvement_pct": ((rmse_0shot - rmse_60min) / max(rmse_0shot, 1e-6)) * 100.0,
            }
        )

    results = pd.DataFrame(rows).sort_values("rmse_0shot", ascending=False).reset_index(drop=True)
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print()
    print("Cross-archetype evaluation")
    print("-" * 84)
    print(results.to_string(index=False, formatters={
        "rmse_0shot": "{:.3f}".format,
        "rmse_30min": "{:.3f}".format,
        "rmse_60min": "{:.3f}".format,
        "rmse_improvement_pct": "{:.2f}".format,
    }))
    print("-" * 84)
    print()

    return results


__all__ = ["cross_archetype_evaluation", "plot_adaptation_curve"]

