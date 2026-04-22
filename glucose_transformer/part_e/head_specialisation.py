"""Attention-head specialisation analysis for Part E."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from part_d.cohort_simulator import ECG_COLUMNS, EMG_COLUMNS
from part_e.common import make_window_batch, save_json, trace_model


def _head_metrics(attention_matrix: np.ndarray) -> dict[str, float]:
    """Characterise one head by distance, entropy, and peak position."""

    matrix = np.asarray(attention_matrix, dtype=np.float32)
    matrix = matrix / np.clip(matrix.sum(axis=-1, keepdims=True), 1e-6, None)
    seq_len = matrix.shape[0]
    distances = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]).astype("float32")
    mean_attention = matrix.mean(axis=0)
    attention_distance = float((matrix * distances).sum() / max(matrix.sum(), 1e-6))
    attention_entropy = float(-(matrix * np.log(np.clip(matrix, 1e-6, None))).sum() / matrix.size)
    top_position = int(np.argmax(mean_attention))
    return {
        "attention_distance": attention_distance,
        "attention_entropy": attention_entropy,
        "top_position": top_position,
    }


def _label_head(metrics: dict[str, float]) -> str:
    """Assign an informal functional label to an attention head."""

    distance = metrics["attention_distance"]
    entropy = metrics["attention_entropy"]
    top_position = metrics["top_position"]

    if distance < 2.0 and entropy > 0.10:
        return "Short-range smoothing"
    if 6 <= top_position <= 9:
        return "Physiological lag detector"
    if 1 <= top_position <= 3:
        return "Recent trend tracker"
    return "Global context aggregator"


def analyse_head_specialisation(model, test_loader, *, config: dict) -> dict:
    """Compute average per-head attention patterns and characterise them."""

    max_windows = int(config["head_analysis_max_windows"])
    batch_size = int(config["attention_batch_size"])
    device = str(next(model.parameters()).device)

    window_entries = []
    for window_entry in test_loader.iter_split_windows(
        "test",
        max_windows_per_user=int(config["analysis_max_windows_per_user"]),
    ):
        window_entries.append(window_entry)
        if len(window_entries) >= max_windows:
            break

    hr_layer_accumulators: list[list[np.ndarray]] | None = None
    ecg_feature_signal = np.zeros(len(ECG_COLUMNS), dtype=np.float64)
    emg_feature_signal = np.zeros(len(EMG_COLUMNS), dtype=np.float64)
    n_batches = 0

    for start in range(0, len(window_entries), batch_size):
        batch_entries = window_entries[start:start + batch_size]
        batch = make_window_batch(batch_entries, device=device)
        trace = trace_model(model, batch, capture_attention=True)

        if hr_layer_accumulators is None:
            hr_layer_accumulators = [[] for _ in range(len(trace["hr"]["attention_weights"]))]

        for layer_index, attention in enumerate(trace["hr"]["attention_weights"]):
            attention_np = attention.numpy()[:, :, 1:, 1:]
            hr_layer_accumulators[layer_index].append(attention_np.mean(axis=0))

        hr_to_ecg = trace["cross_attention"]["hr_to_ecg"].detach().cpu().numpy()[:, :, 1:, 1:]
        hr_to_emg = trace["cross_attention"]["hr_to_emg"].detach().cpu().numpy()[:, :, 1:, 1:]
        ecg_token_weights = hr_to_ecg.mean(axis=1).mean(axis=1)  # [batch, seq]
        emg_token_weights = hr_to_emg.mean(axis=1).mean(axis=1)

        for batch_index, entry in enumerate(batch_entries):
            raw_ecg = np.asarray(entry["metadata"]["raw"]["ecg_features"], dtype=np.float32)
            raw_emg = np.asarray(entry["metadata"]["raw"]["emg_features"], dtype=np.float32)
            ecg_feature_signal += (np.abs(raw_ecg) * ecg_token_weights[batch_index][:, None]).sum(axis=0)
            emg_feature_signal += (np.abs(raw_emg) * emg_token_weights[batch_index][:, None]).sum(axis=0)

        n_batches += 1

    if hr_layer_accumulators is None:
        raise ValueError("No test windows were available for head analysis.")

    per_head_rows: list[dict[str, Any]] = []
    metrics_table = []
    row_labels = []
    for layer_index, layer_batches in enumerate(hr_layer_accumulators):
        mean_by_head = np.mean(layer_batches, axis=0)  # [heads, seq, seq]
        for head_index in range(mean_by_head.shape[0]):
            metrics = _head_metrics(mean_by_head[head_index])
            label = _label_head(metrics)
            per_head_rows.append(
                {
                    "layer": int(layer_index + 1),
                    "head": int(head_index + 1),
                    **metrics,
                    "label": label,
                }
            )
            metrics_table.append(
                [
                    metrics["attention_distance"],
                    metrics["attention_entropy"],
                    metrics["top_position"],
                ]
            )
            row_labels.append(f"L{layer_index + 1}H{head_index + 1}")

    ecg_feature_importance = (ecg_feature_signal / np.clip(ecg_feature_signal.sum(), 1e-6, None)).astype("float32")
    emg_feature_importance = (emg_feature_signal / np.clip(emg_feature_signal.sum(), 1e-6, None)).astype("float32")

    output_path = Path(config["head_specialisation_plot_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14, 7))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.7, 1.0])

    axis_metrics = figure.add_subplot(grid[0, 0])
    metric_image = axis_metrics.imshow(np.asarray(metrics_table, dtype=np.float32), aspect="auto", cmap="coolwarm")
    axis_metrics.set_yticks(np.arange(len(row_labels)))
    axis_metrics.set_yticklabels(row_labels)
    axis_metrics.set_xticks([0, 1, 2])
    axis_metrics.set_xticklabels(["Distance", "Entropy", "Top Pos"])
    axis_metrics.set_title("HR Encoder Head Characterisation")
    figure.colorbar(metric_image, ax=axis_metrics, fraction=0.046, pad=0.04)

    axis_features = figure.add_subplot(grid[0, 1])
    x_ecg = np.arange(len(ECG_COLUMNS))
    x_emg = np.arange(len(EMG_COLUMNS)) + len(ECG_COLUMNS) + 1
    axis_features.bar(x_ecg, ecg_feature_importance, label="ECG", color="#3182bd")
    axis_features.bar(x_emg, emg_feature_importance, label="EMG", color="#31a354")
    axis_features.set_xticks(list(x_ecg) + list(x_emg))
    axis_features.set_xticklabels(ECG_COLUMNS + EMG_COLUMNS, rotation=35, ha="right")
    axis_features.set_ylabel("Attention-weighted feature salience")
    axis_features.set_title("Cross-Attention Feature Salience")
    axis_features.legend()
    axis_features.grid(axis="y", alpha=0.22)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    payload = {
        "per_head": per_head_rows,
        "ecg_feature_importance": {
            column: float(value) for column, value in zip(ECG_COLUMNS, ecg_feature_importance)
        },
        "emg_feature_importance": {
            column: float(value) for column, value in zip(EMG_COLUMNS, emg_feature_importance)
        },
        "plot_path": str(output_path),
        "n_windows": len(window_entries),
        "n_batches": n_batches,
    }
    save_json(payload, config["head_specialisation_results_path"])
    return payload


__all__ = ["analyse_head_specialisation"]
