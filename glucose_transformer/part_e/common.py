"""Shared helpers for Part E analyses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from part_d.dataset import MetaLearningDataset
from part_d.maml import load_meta_checkpoint
from part_d.user_embedding import UserConditionedFullModalTransformer


def ensure_runtime_dirs(config: dict) -> None:
    """Create all Part E output directories."""

    for key in ["results_dir_part_e", "figures_dir_part_e", "checkpoint_dir_part_e", "ig_scenarios_dir"]:
        Path(config[key]).mkdir(parents=True, exist_ok=True)


def save_json(payload: dict | list, output_path: str | Path) -> None:
    """Persist a JSON-serialisable object to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(input_path: str | Path) -> dict | list:
    """Load a JSON file from disk."""

    return json.loads(Path(input_path).read_text(encoding="utf-8"))


def load_part_d_summary(config: dict) -> dict:
    """Load the Part D summary when available."""

    summary_path = Path(config["part_d_summary_path"])
    return load_json(summary_path) if summary_path.exists() else {}


def infer_backbone_kind(config: dict) -> str:
    """Infer which EEG backbone was used for the trained Part D model."""

    summary = load_part_d_summary(config)
    backbone_kind = summary.get("backbone_kind")
    if backbone_kind in {"frequency_eeg", "patch_eeg", "hierarchical_eeg"}:
        return str(backbone_kind)
    return str(config["part_d_eeg_encoder_kind"])


def load_model_and_dataset(config: dict) -> tuple[UserConditionedFullModalTransformer, MetaLearningDataset]:
    """Load the trained Part D model and the synthetic cohort dataset."""

    dataset = MetaLearningDataset(config)
    backbone_kind = infer_backbone_kind(config)
    model = UserConditionedFullModalTransformer(
        config,
        eeg_encoder_kind=backbone_kind,
        n_users=len(dataset.manifest),
        inject_conditioning=True,
    ).to(config["device"])
    model.set_known_user_ids(dataset.get_known_user_ids())
    load_meta_checkpoint(model, config["best_meta_checkpoint_path"], device=config["device"])
    model.eval()
    return model, dataset


def make_single_window_batch(window_entry: dict, *, device: str) -> dict[str, Any]:
    """Convert a window entry from `MetaLearningDataset.iter_split_windows()` into a model batch."""

    sample = window_entry["sample"]
    metadata = window_entry["metadata"]
    return {
        "hr_sequence": sample["hr_sequence"].unsqueeze(0).to(device),
        "glucose_context": sample["glucose_context"].unsqueeze(0).to(device),
        "ecg_features": sample["ecg_features"].unsqueeze(0).to(device),
        "emg_features": sample["emg_features"].unsqueeze(0).to(device),
        "eeg_signal": sample["eeg_signal"].unsqueeze(0).to(device),
        "cbf_signal": sample["cbf_signal"].unsqueeze(0).to(device),
        "targets": sample["targets"].unsqueeze(0).to(device),
        "user_ids": torch.tensor([int(metadata["user_id"])], dtype=torch.long, device=device),
        "archetype_ids": torch.tensor([int(metadata["archetype_id"])], dtype=torch.long, device=device),
        "window_entry": window_entry,
    }


def make_window_batch(window_entries: list[dict], *, device: str) -> dict[str, Any]:
    """Convert multiple window entries into one batched model input."""

    samples = [entry["sample"] for entry in window_entries]
    metadata = [entry["metadata"] for entry in window_entries]
    return {
        "hr_sequence": torch.stack([sample["hr_sequence"] for sample in samples], dim=0).to(device),
        "glucose_context": torch.stack([sample["glucose_context"] for sample in samples], dim=0).to(device),
        "ecg_features": torch.stack([sample["ecg_features"] for sample in samples], dim=0).to(device),
        "emg_features": torch.stack([sample["emg_features"] for sample in samples], dim=0).to(device),
        "eeg_signal": torch.stack([sample["eeg_signal"] for sample in samples], dim=0).to(device),
        "cbf_signal": torch.stack([sample["cbf_signal"] for sample in samples], dim=0).to(device),
        "targets": torch.stack([sample["targets"] for sample in samples], dim=0).to(device),
        "user_ids": torch.tensor([int(item["user_id"]) for item in metadata], dtype=torch.long, device=device),
        "archetype_ids": torch.tensor([int(item["archetype_id"]) for item in metadata], dtype=torch.long, device=device),
        "window_entries": window_entries,
    }


def normalise_batch(batch: dict[str, Any], *, device: str) -> dict[str, Any]:
    """Move a Part E batch dictionary onto the requested device."""

    tensor_keys = [
        "hr_sequence",
        "glucose_context",
        "ecg_features",
        "emg_features",
        "eeg_signal",
        "cbf_signal",
        "targets",
        "user_ids",
        "archetype_ids",
    ]
    normalised = dict(batch)
    for key in tensor_keys:
        if key in normalised and isinstance(normalised[key], torch.Tensor):
            normalised[key] = normalised[key].to(device)
    return normalised


def _run_transformer_stack(
    encoder: torch.nn.TransformerEncoder,
    hidden: torch.Tensor,
    *,
    capture_attention: bool,
) -> dict[str, Any]:
    """Run a transformer stack layer by layer and retain intermediate states."""

    layer_outputs: list[torch.Tensor] = []
    attention_weights: list[torch.Tensor] = []
    for layer in encoder.layers:
        if hasattr(layer, "capture_attention"):
            layer.capture_attention = capture_attention
        hidden = layer(hidden)
        layer_outputs.append(hidden)
        if capture_attention and getattr(layer, "latest_attention_weights", None) is not None:
            attention_weights.append(layer.latest_attention_weights.detach().cpu())

    if encoder.norm is not None:
        hidden = encoder.norm(hidden)
        if layer_outputs:
            layer_outputs[-1] = hidden

    return {
        "output": hidden,
        "layer_outputs": layer_outputs,
        "attention_weights": attention_weights,
    }


def _trace_sequence_encoder(module, inputs: torch.Tensor, *, capture_attention: bool) -> dict[str, Any]:
    """Trace a `SequenceEncoder` or `TrackedSequenceEncoder`-style module."""

    batch_size = inputs.size(0)
    hidden = module.token_embedding(inputs)
    if getattr(module, "modality_embedding", None) is not None:
        hidden = hidden + module.modality_embedding

    use_cls = getattr(module, "use_cls_token", True)
    cls_token = getattr(module, "cls_token", None)
    if use_cls and cls_token is not None:
        hidden = torch.cat([cls_token.expand(batch_size, -1, -1), hidden], dim=1)

    positioned = module.position_encoder(hidden)
    stack_trace = _run_transformer_stack(
        module.encoder,
        positioned,
        capture_attention=capture_attention,
    )
    stack_trace["input_tokens"] = positioned
    stack_trace["uses_cls"] = bool(use_cls and cls_token is not None)
    return stack_trace


def _trace_eeg_encoder(eeg_encoder, eeg_signal: torch.Tensor, *, capture_attention: bool) -> dict[str, Any]:
    """Trace the selected Part C EEG encoder."""

    kind = type(eeg_encoder).__name__
    if hasattr(eeg_encoder, "_band_power_tokens"):
        tokens = eeg_encoder._band_power_tokens(eeg_signal)
        trace = _trace_sequence_encoder(eeg_encoder.encoder, tokens, capture_attention=capture_attention)
        summary = trace["output"].mean(dim=1)
        return {
            "kind": "frequency_eeg",
            "tokens": tokens,
            "summary": summary,
            "trace": trace,
        }

    if hasattr(eeg_encoder, "_patchify"):
        tokens = eeg_encoder._patchify(eeg_signal)
        trace = _trace_sequence_encoder(eeg_encoder.encoder, tokens, capture_attention=capture_attention)
        summary = trace["output"].mean(dim=1)
        return {
            "kind": "patch_eeg",
            "tokens": tokens,
            "summary": summary,
            "trace": trace,
        }

    if hasattr(eeg_encoder, "_local_summaries"):
        local_summaries = eeg_encoder._local_summaries(eeg_signal)
        trace = _trace_sequence_encoder(
            eeg_encoder.global_encoder.encoder,
            local_summaries,
            capture_attention=capture_attention,
        )
        summary = trace["output"][:, 0, :]
        return {
            "kind": "hierarchical_eeg",
            "tokens": local_summaries,
            "summary": summary,
            "trace": trace,
        }

    raise ValueError(f"Unsupported EEG encoder for tracing: {kind}")


def trace_model(model: UserConditionedFullModalTransformer, batch: dict[str, Any], *, capture_attention: bool) -> dict[str, Any]:
    """Run the Part D model manually and keep intermediate activations."""

    backbone = model.backbone
    batch = normalise_batch(batch, device=batch["hr_sequence"].device)
    user_embedding = model.resolve_user_embedding(
        batch_size=batch["hr_sequence"].size(0),
        user_ids=batch["user_ids"],
        archetype_ids=batch["archetype_ids"],
        device=batch["hr_sequence"].device,
    )
    model.conditioning_context.current_embedding = user_embedding

    try:
        eeg_trace = _trace_eeg_encoder(backbone.eeg_encoder, batch["eeg_signal"], capture_attention=capture_attention)

        hr_inputs = torch.cat([batch["hr_sequence"], batch["glucose_context"]], dim=-1)
        hr_trace = _trace_sequence_encoder(backbone.hr_encoder, hr_inputs, capture_attention=capture_attention)
        ecg_trace = _trace_sequence_encoder(backbone.ecg_encoder, batch["ecg_features"], capture_attention=capture_attention)
        emg_trace = _trace_sequence_encoder(backbone.emg_encoder, batch["emg_features"], capture_attention=capture_attention)
        cbf_trace = _trace_sequence_encoder(backbone.cbf_encoder, batch["cbf_signal"], capture_attention=capture_attention)

        hr_enriched_with_ecg, hr_to_ecg_weights = backbone.hr_to_ecg_attention(
            query=hr_trace["output"],
            key=ecg_trace["output"],
            value=ecg_trace["output"],
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        hr_enriched_with_emg, hr_to_emg_weights = backbone.hr_to_emg_attention(
            query=hr_trace["output"],
            key=emg_trace["output"],
            value=emg_trace["output"],
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        cbf_summary = cbf_trace["output"][:, 0, :]
        cbf_context = cbf_summary.unsqueeze(1).expand(-1, hr_trace["output"].size(1), -1)
        hr_enriched_with_cbf, hr_to_cbf_weights = backbone.hr_to_cbf_attention(
            query=hr_trace["output"],
            key=cbf_context,
            value=cbf_context,
            need_weights=capture_attention,
            average_attn_weights=False,
        )

        hr_fused = backbone.fusion_norm(
            hr_trace["output"] + hr_enriched_with_ecg + hr_enriched_with_emg + hr_enriched_with_cbf
        )
        eeg_token = backbone.eeg_summary_projection(eeg_trace["summary"]).unsqueeze(1)
        fused_sequence = torch.cat([hr_fused, eeg_token], dim=1)
        final_trace = _run_transformer_stack(
            backbone.final_fusion_encoder,
            fused_sequence,
            capture_attention=capture_attention,
        )
        final_trace["input_tokens"] = fused_sequence
        predictions = backbone.regression_head(final_trace["output"][:, 0, :])
    finally:
        model.conditioning_context.current_embedding = None

    return {
        "predictions": predictions,
        "user_embedding": user_embedding,
        "eeg": eeg_trace,
        "hr": hr_trace,
        "ecg": ecg_trace,
        "emg": emg_trace,
        "cbf": cbf_trace,
        "cross_attention": {
            "hr_to_ecg": hr_to_ecg_weights,
            "hr_to_emg": hr_to_emg_weights,
            "hr_to_cbf": hr_to_cbf_weights,
        },
        "hr_fused": hr_fused,
        "final": final_trace,
    }


def build_analysis_windows(
    dataset: MetaLearningDataset,
    *,
    split: str,
    limit_users: int | None = None,
    max_windows_per_user: int | None = None,
) -> list[dict[str, Any]]:
    """Materialise a deterministic list of window entries for one split."""

    return list(
        dataset.iter_split_windows(
            split,
            limit_users=limit_users,
            max_windows_per_user=max_windows_per_user,
        )
    )
