"""Attention visualisation helpers for the Part A Transformer."""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from part_a.config import CONFIG


def _to_numpy_attention(attention_weights: list[torch.Tensor]) -> list[np.ndarray]:
    """Convert attention tensors to NumPy arrays on CPU."""

    return [tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor) for tensor in attention_weights]


def plot_attention_heatmap(attention_weights, layer, head, save_path):
    """Plot and save the attention matrix for one layer/head selection."""

    attention_arrays = _to_numpy_attention(attention_weights)
    attention_matrix = attention_arrays[layer][0, head]

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(attention_matrix, cmap=plt.cm.RdBu_r, aspect="auto")
    axis.set_title(f"Layer {layer}, Head {head} Attention Weights")
    axis.set_xlabel("Key (time steps)")
    axis.set_ylabel("Query (time steps)")
    axis.set_xticks(range(attention_matrix.shape[-1]))
    axis.set_yticks(range(attention_matrix.shape[-2]))
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close(figure)


def _attention_rollout(attention_weights: list[np.ndarray]) -> np.ndarray:
    """Aggregate multi-layer attention with residual-aware rollout."""

    layer_averages = [layer_weights[0].mean(axis=0) for layer_weights in attention_weights]
    sequence_length = layer_averages[0].shape[-1]
    rollout = np.eye(sequence_length, dtype=np.float32)

    for layer_attention in layer_averages:
        augmented_attention = layer_attention + np.eye(sequence_length, dtype=np.float32)
        augmented_attention = augmented_attention / augmented_attention.sum(axis=-1, keepdims=True)
        rollout = augmented_attention @ rollout

    return rollout


def plot_temporal_attention_profile(attention_weights, save_path):
    """Plot a rollout-style temporal profile of CLS attention over the input window."""

    attention_arrays = _to_numpy_attention(attention_weights)
    rollout = _attention_rollout(attention_arrays)
    cls_to_timesteps = rollout[0, 1:]

    # Reverse to express the x-axis as minutes before prediction: 0 is most recent.
    profile = cls_to_timesteps[::-1]
    minutes_before_prediction = np.arange(profile.shape[0]) * 5

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(minutes_before_prediction, profile, color="#1f77b4", linewidth=2.0)
    axis.fill_between(minutes_before_prediction, profile, alpha=0.15, color="#1f77b4")
    axis.axvspan(
        CONFIG["lag_zone_start_minutes"],
        CONFIG["lag_zone_end_minutes"],
        color="#ffcc80",
        alpha=0.35,
        label="expected glucose-HR lag zone",
    )
    axis.set_xlabel("Minutes Before Prediction")
    axis.set_ylabel("Average Attention Weight")
    axis.set_title("Temporal Attention Profile")
    axis.legend(loc="upper right")
    axis.grid(alpha=0.25)
    figure.tight_layout()

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=CONFIG["figure_dpi"], bbox_inches="tight")
    plt.close(figure)


def save_random_attention_visualisations(
    model: torch.nn.Module,
    dataset,
    *,
    device: str,
    output_dir: str | Path,
    sample_count: int,
) -> list[dict]:
    """Generate heatmaps and temporal profiles for random held-out windows."""

    if len(dataset) == 0:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    selected_indices = random.sample(range(len(dataset)), k=min(sample_count, len(dataset)))
    saved_artifacts: list[dict] = []

    for index in selected_indices:
        hr_sequence, glucose_context, _ = dataset[index]
        metadata = dataset.get_metadata(index)

        attention_weights = model.get_attention_weights(
            hr_sequence.unsqueeze(0).to(device),
            glucose_context.unsqueeze(0).to(device),
        )

        filename_stem = (
            f"patient_{metadata['patient_id']}_"
            f"{metadata['timestamp'].strftime('%Y%m%d_%H%M')}_"
            f"sample_{index}"
        )
        heatmap_path = output_path / f"{filename_stem}_heatmap.png"
        profile_path = output_path / f"{filename_stem}_profile.png"

        plot_attention_heatmap(attention_weights, layer=0, head=0, save_path=heatmap_path)
        plot_temporal_attention_profile(attention_weights, save_path=profile_path)

        saved_artifacts.append(
            {
                "index": index,
                "patient_id": int(metadata["patient_id"]),
                "heatmap_path": str(heatmap_path),
                "profile_path": str(profile_path),
            }
        )

    return saved_artifacts


__all__ = [
    "plot_attention_heatmap",
    "plot_temporal_attention_profile",
    "save_random_attention_visualisations",
]
