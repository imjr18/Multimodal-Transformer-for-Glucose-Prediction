"""Attention-rollout analysis for Part E."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from part_c.models.common import resample_profile
from part_e.common import normalise_batch, trace_model


def _average_attention(attention: torch.Tensor) -> torch.Tensor:
    """Average a `[batch, heads, q, k]` attention tensor into `[q, k]`."""

    return attention.detach().cpu().float().mean(dim=0).mean(dim=0)


def _row_normalise(matrix: torch.Tensor) -> torch.Tensor:
    """Normalise an attention matrix so each row sums to one."""

    return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _rollout_matrix(attention_weights: list[torch.Tensor], *, residual_weight: float = 0.5) -> torch.Tensor:
    """Compose a list of self-attention maps into one effective attention matrix."""

    if not attention_weights:
        raise ValueError("No attention weights were provided for rollout.")

    sequence_length = int(attention_weights[0].shape[-1])
    rollout = torch.eye(sequence_length, dtype=torch.float32)
    identity = torch.eye(sequence_length, dtype=torch.float32)

    for attention in attention_weights:
        averaged = _average_attention(attention)
        augmented = (residual_weight * averaged) + ((1.0 - residual_weight) * identity)
        rollout = _row_normalise(augmented) @ rollout
    return rollout


def _cross_attention_matrix(attention: torch.Tensor) -> torch.Tensor:
    """Collapse one cross-attention tensor into `[query, key]`."""

    return _row_normalise(_average_attention(attention))


def _profile_from_rollout(rollout: torch.Tensor, *, has_cls_token: bool) -> np.ndarray:
    """Reduce a rollout matrix to a one-dimensional importance profile."""

    if has_cls_token:
        profile = rollout[0, 1:]
    else:
        profile = rollout.mean(dim=0)
    profile = profile / profile.sum().clamp_min(1e-6)
    return profile.numpy().astype("float32")


def _normalise_profile(profile: np.ndarray) -> np.ndarray:
    """Normalise a 1D profile so it sums to one."""

    profile = np.asarray(profile, dtype=np.float32).reshape(-1)
    total = float(np.clip(profile.sum(), 1e-6, None))
    return (profile / total).astype("float32")


def compute_attention_rollout(model, sample_batch) -> dict:
    """Compute per-modality temporal importance via attention rollout."""

    device = next(model.parameters()).device
    batch = normalise_batch(sample_batch, device=str(device))
    trace = trace_model(model, batch, capture_attention=True)

    final_rollout = _rollout_matrix(trace["final"]["attention_weights"])
    hr_rollout = _rollout_matrix(trace["hr"]["attention_weights"])
    ecg_rollout = _rollout_matrix(trace["ecg"]["attention_weights"])
    emg_rollout = _rollout_matrix(trace["emg"]["attention_weights"])
    cbf_rollout = _rollout_matrix(trace["cbf"]["attention_weights"])

    final_to_hr_branch = final_rollout[0, : hr_rollout.shape[0]]
    hr_temporal = _normalise_profile((final_to_hr_branch @ hr_rollout)[1:].numpy())

    hr_to_ecg = _cross_attention_matrix(trace["cross_attention"]["hr_to_ecg"])
    ecg_temporal = _normalise_profile((final_to_hr_branch @ hr_to_ecg @ ecg_rollout)[1:].numpy())

    hr_to_emg = _cross_attention_matrix(trace["cross_attention"]["hr_to_emg"])
    emg_temporal = _normalise_profile((final_to_hr_branch @ hr_to_emg @ emg_rollout)[1:].numpy())

    hr_to_cbf = _cross_attention_matrix(trace["cross_attention"]["hr_to_cbf"])
    cbf_temporal = _normalise_profile((final_to_hr_branch @ hr_to_cbf @ cbf_rollout)[1:].numpy())

    eeg_attention_weights = trace["eeg"]["trace"]["attention_weights"]
    if eeg_attention_weights:
        eeg_rollout = _rollout_matrix(eeg_attention_weights)
        eeg_profile = _profile_from_rollout(
            eeg_rollout,
            has_cls_token=bool(trace["eeg"]["trace"]["uses_cls"]),
        )
    else:
        token_count = int(trace["eeg"]["tokens"].shape[1])
        eeg_profile = np.full(token_count, 1.0 / max(token_count, 1), dtype=np.float32)

    completeness = {
        "hr_temporal_importance": float(hr_temporal.sum()),
        "ecg_temporal_importance": float(ecg_temporal.sum()),
        "emg_temporal_importance": float(emg_temporal.sum()),
        "eeg_patch_importance": float(eeg_profile.sum()),
        "cbf_temporal_importance": float(cbf_temporal.sum()),
    }

    return {
        "hr_temporal_importance": hr_temporal,
        "ecg_temporal_importance": ecg_temporal,
        "emg_temporal_importance": emg_temporal,
        "eeg_patch_importance": eeg_profile,
        "cbf_temporal_importance": cbf_temporal,
        "completeness": completeness,
    }


def plot_temporal_importance_profile(rollout_dict: dict[str, Any], save_path: str | Path) -> str:
    """Plot rollout-derived temporal importance profiles for all modalities."""

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shared_minutes = np.arange(24, dtype=np.float32)[::-1] * 5.0
    modality_profiles = [
        ("HR", rollout_dict["hr_temporal_importance"], (30.0, 45.0), "#d7301f"),
        ("ECG", rollout_dict["ecg_temporal_importance"], (15.0, 30.0), "#3182bd"),
        ("EMG", rollout_dict["emg_temporal_importance"], (30.0, 60.0), "#31a354"),
        ("EEG", rollout_dict["eeg_patch_importance"], (60.0, 120.0), "#756bb1"),
        ("CBF", rollout_dict["cbf_temporal_importance"], (30.0, 90.0), "#636363"),
    ]

    figure, axes = plt.subplots(len(modality_profiles), 1, figsize=(10, 11), sharex=True)
    for axis, (label, profile, lag_zone, color) in zip(axes, modality_profiles):
        plot_profile = resample_profile(np.asarray(profile, dtype=np.float32), target_length=24)
        axis.plot(shared_minutes, plot_profile, color=color, linewidth=2.0)
        axis.fill_between(shared_minutes, 0.0, plot_profile, color=color, alpha=0.12)
        axis.axvspan(lag_zone[0], lag_zone[1], color=color, alpha=0.15)
        axis.text(
            lag_zone[0] + 1.0,
            max(float(plot_profile.max()) * 0.85, 0.02),
            "expected lag zone",
            fontsize=9,
            color=color,
        )
        axis.set_ylabel("Importance")
        axis.set_title(label)
        axis.grid(alpha=0.22)

    axes[-1].set_xlabel("Minutes Before Prediction (0 = most recent)")
    axes[-1].invert_xaxis()
    figure.suptitle("Attention Rollout Temporal Importance", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return str(output_path)


__all__ = ["compute_attention_rollout", "plot_temporal_importance_profile"]
