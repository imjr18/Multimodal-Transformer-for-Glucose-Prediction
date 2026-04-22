"""Spurious-correlation control with an auxiliary noise modality."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from part_b.models.cross_attention import ModalityEncoder
from part_d.user_embedding import UserConditionedEncoderLayer
from part_e.common import load_json, save_json


class NoiseWindowDataset(Dataset):
    """Dataset of Part D windows extended with deterministic Gaussian noise."""

    def __init__(self, window_entries: list[dict[str, Any]], *, base_seed: int):
        self.window_entries = list(window_entries)
        self.base_seed = int(base_seed)

    def __len__(self) -> int:
        return len(self.window_entries)

    def __getitem__(self, index: int):
        entry = self.window_entries[index]
        sample = entry["sample"]
        metadata = entry["metadata"]
        rng = np.random.default_rng(self.base_seed + int(metadata["user_id"]) * 1009 + int(metadata["start_idx"]))
        noise = torch.tensor(
            rng.normal(0.0, 1.0, size=(sample["hr_sequence"].shape[0], 2)).astype("float32"),
            dtype=torch.float32,
        )
        return (
            sample["hr_sequence"],
            sample["glucose_context"],
            sample["ecg_features"],
            sample["emg_features"],
            sample["eeg_signal"],
            sample["cbf_signal"],
            noise,
            sample["targets"],
            int(metadata["user_id"]),
            int(metadata["archetype_id"]),
        )


class NoiseAwareModel(nn.Module):
    """Part D backbone plus one auxiliary noise-modality branch."""

    def __init__(self, base_model, config: dict, *, zero_init_noise: bool = False):
        super().__init__()
        self.base_model = deepcopy(base_model)
        self.config = config
        self.noise_encoder = ModalityEncoder(n_features=2, config=config)
        self.hr_to_noise_attention = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self._wrap_noise_encoder()
        if zero_init_noise:
            self._zero_initialise_noise_branch()

    def _wrap_noise_encoder(self) -> None:
        """Apply the Part D user-conditioning wrapper to the noise encoder."""

        encoder = self.noise_encoder.encoder
        if isinstance(encoder.layers[0], UserConditionedEncoderLayer):
            return
        d_model = int(self.config["d_model"])
        context = self.base_model.conditioning_context
        encoder.layers = nn.ModuleList(
            [
                UserConditionedEncoderLayer(
                    layer,
                    context,
                    embedding_dim=int(self.config["user_embedding_dim"]),
                    d_model=d_model,
                )
                for layer in encoder.layers
            ]
        )

    def _zero_initialise_noise_branch(self) -> None:
        """Make the noise branch a structural no-op for the control model."""

        for parameter in self.noise_encoder.parameters():
            nn.init.zeros_(parameter)
        for parameter in self.hr_to_noise_attention.parameters():
            nn.init.zeros_(parameter)

    def forward(
        self,
        hr_sequence: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        eeg_signal: torch.Tensor,
        cbf_signal: torch.Tensor,
        noise_features: torch.Tensor,
        *,
        user_ids: torch.Tensor,
        archetype_ids: torch.Tensor,
        capture_noise_attention: bool = False,
    ):
        """Run the noise-augmented forward pass."""

        base_model = self.base_model
        backbone = base_model.backbone
        user_embedding = base_model.resolve_user_embedding(
            batch_size=hr_sequence.size(0),
            user_ids=user_ids,
            archetype_ids=archetype_ids,
            device=hr_sequence.device,
        )
        base_model.conditioning_context.current_embedding = user_embedding
        try:
            eeg_summary = backbone.eeg_encoder(eeg_signal)
            hr_inputs = torch.cat([hr_sequence, glucose_context], dim=-1)
            hr_encoded = backbone.hr_encoder(hr_inputs)
            ecg_encoded = backbone.ecg_encoder(ecg_features)
            emg_encoded = backbone.emg_encoder(emg_features)
            cbf_encoded = backbone.cbf_encoder(cbf_signal)
            noise_encoded = self.noise_encoder(noise_features)

            hr_enriched_ecg, _ = backbone.hr_to_ecg_attention(
                query=hr_encoded,
                key=ecg_encoded,
                value=ecg_encoded,
                need_weights=False,
                average_attn_weights=False,
            )
            hr_enriched_emg, _ = backbone.hr_to_emg_attention(
                query=hr_encoded,
                key=emg_encoded,
                value=emg_encoded,
                need_weights=False,
                average_attn_weights=False,
            )
            cbf_summary = cbf_encoded[:, 0, :]
            cbf_context = cbf_summary.unsqueeze(1).expand(-1, hr_encoded.size(1), -1)
            hr_enriched_cbf, _ = backbone.hr_to_cbf_attention(
                query=hr_encoded,
                key=cbf_context,
                value=cbf_context,
                need_weights=False,
                average_attn_weights=False,
            )
            hr_enriched_noise, hr_to_noise_weights = self.hr_to_noise_attention(
                query=hr_encoded,
                key=noise_encoded,
                value=noise_encoded,
                need_weights=capture_noise_attention,
                average_attn_weights=False,
            )

            hr_fused = backbone.fusion_norm(
                hr_encoded + hr_enriched_ecg + hr_enriched_emg + hr_enriched_cbf + hr_enriched_noise
            )
            eeg_token = backbone.eeg_summary_projection(eeg_summary).unsqueeze(1)
            fused_sequence = torch.cat([hr_fused, eeg_token], dim=1)
            fused_encoded = backbone.final_fusion_encoder(fused_sequence)
            predictions = backbone.regression_head(fused_encoded[:, 0, :])

            noise_share = (
                torch.mean(torch.abs(hr_enriched_noise))
                / (
                    torch.mean(torch.abs(hr_enriched_ecg))
                    + torch.mean(torch.abs(hr_enriched_emg))
                    + torch.mean(torch.abs(hr_enriched_cbf))
                    + torch.mean(torch.abs(hr_enriched_noise))
                    + 1e-6
                )
            )
        finally:
            base_model.conditioning_context.current_embedding = None

        return predictions, hr_to_noise_weights, noise_share


def _collate_to_device(batch, *, device: str):
    """Move a noise-window batch to the target device."""

    tensors = []
    for item in batch:
        if isinstance(item, torch.Tensor):
            tensors.append(item.to(device))
        else:
            tensors.append(torch.tensor(item, dtype=torch.long, device=device))
    return tensors


def _train_noise_model(model: NoiseAwareModel, train_loader, val_loader, *, config: dict) -> None:
    """Train the noise-aware control model with simple supervised loss."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["spurious_learning_rate"]),
        betas=config["adam_betas"],
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    checkpoint_path = Path(config["spurious_checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config["spurious_epochs"]) + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Noise Epoch {epoch}", leave=False):
            (
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                noise_features,
                targets,
                user_ids,
                archetype_ids,
            ) = _collate_to_device(batch, device=str(next(model.parameters()).device))

            optimizer.zero_grad(set_to_none=True)
            predictions, _, _ = model(
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                noise_features,
                user_ids=user_ids,
                archetype_ids=archetype_ids,
            )
            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(config["grad_clip_norm"]))
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                (
                    hr_sequence,
                    glucose_context,
                    ecg_features,
                    emg_features,
                    eeg_signal,
                    cbf_signal,
                    noise_features,
                    targets,
                    user_ids,
                    archetype_ids,
                ) = _collate_to_device(batch, device=str(next(model.parameters()).device))
                predictions, _, _ = model(
                    hr_sequence,
                    glucose_context,
                    ecg_features,
                    emg_features,
                    eeg_signal,
                    cbf_signal,
                    noise_features,
                    user_ids=user_ids,
                    archetype_ids=archetype_ids,
                )
                val_losses.append(float(criterion(predictions, targets).item()))

        mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model.state_dict(), checkpoint_path)


def _noise_integrated_gradients(model: NoiseAwareModel, batch, *, device: str, n_steps: int) -> tuple[float, float]:
    """Compute the percentage of total IG attribution assigned to the noise branch."""

    (
        hr_sequence,
        glucose_context,
        ecg_features,
        emg_features,
        eeg_signal,
        cbf_signal,
        noise_features,
        targets,
        user_ids,
        archetype_ids,
    ) = batch

    input_tensors = {
        "hr_sequence": hr_sequence.to(device).detach().clone().float(),
        "glucose_context": glucose_context.to(device).detach().clone().float(),
        "ecg_features": ecg_features.to(device).detach().clone().float(),
        "emg_features": emg_features.to(device).detach().clone().float(),
        "eeg_signal": eeg_signal.to(device).detach().clone().float(),
        "cbf_signal": cbf_signal.to(device).detach().clone().float(),
        "noise_features": noise_features.to(device).detach().clone().float(),
    }
    baseline = {key: torch.zeros_like(value) for key, value in input_tensors.items()}
    total_grads = {key: torch.zeros_like(value) for key, value in input_tensors.items()}
    user_ids = user_ids.to(device)
    archetype_ids = archetype_ids.to(device)

    for step in range(1, n_steps + 1):
        alpha = float(step) / float(n_steps)
        scaled_inputs = {}
        for key, value in input_tensors.items():
            scaled = baseline[key] + alpha * (value - baseline[key])
            scaled.requires_grad_(True)
            scaled_inputs[key] = scaled

        predictions, _, _ = model(
            scaled_inputs["hr_sequence"],
            scaled_inputs["glucose_context"],
            scaled_inputs["ecg_features"],
            scaled_inputs["emg_features"],
            scaled_inputs["eeg_signal"],
            scaled_inputs["cbf_signal"],
            scaled_inputs["noise_features"],
            user_ids=user_ids,
            archetype_ids=archetype_ids,
        )
        scalar_output = predictions.mean()
        gradients = torch.autograd.grad(
            scalar_output,
            tuple(scaled_inputs[key] for key in input_tensors),
            retain_graph=False,
            create_graph=False,
        )
        for key, gradient in zip(input_tensors, gradients):
            total_grads[key] += gradient.detach()

    attributions = {
        key: (input_tensors[key] - baseline[key]) * (total_grads[key] / float(n_steps))
        for key in input_tensors
    }
    total_abs = sum(float(torch.sum(torch.abs(value)).item()) for value in attributions.values())
    noise_abs = float(torch.sum(torch.abs(attributions["noise_features"])).item())
    return noise_abs, total_abs


def run_spurious_correlation_test(base_model_path, train_data, test_data, norm_stats, *, config: dict) -> dict:
    """Train an auxiliary noise-aware model and measure spurious reliance on noise."""

    result_path = Path(config["spurious_results_path"])
    if result_path.exists():
        return load_json(result_path)

    train_windows = []
    for entry in train_data.iter_split_windows("train", max_windows_per_user=int(config["analysis_max_windows_per_user"])):
        train_windows.append(entry)
        if len(train_windows) >= int(config["spurious_train_max_windows"]):
            break
    val_windows = []
    for entry in train_data.iter_split_windows("val", max_windows_per_user=int(config["analysis_max_windows_per_user"])):
        val_windows.append(entry)
        if len(val_windows) >= int(config["spurious_val_max_windows"]):
            break
    test_windows = []
    for entry in test_data.iter_split_windows("test", max_windows_per_user=int(config["analysis_max_windows_per_user"])):
        test_windows.append(entry)
        if len(test_windows) >= int(config["spurious_test_max_windows"]):
            break

    train_loader = DataLoader(
        NoiseWindowDataset(train_windows, base_seed=int(config["seed"])),
        batch_size=int(config["spurious_batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        NoiseWindowDataset(val_windows, base_seed=int(config["seed"]) + 17),
        batch_size=int(config["spurious_batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        NoiseWindowDataset(test_windows, base_seed=int(config["seed"]) + 31),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = str(config["device"])
    trained_model = NoiseAwareModel(base_model_path, config).to(device)
    trained_model.base_model.set_known_user_ids(train_data.get_known_user_ids())
    checkpoint_path = Path(config["spurious_checkpoint_path"])
    if checkpoint_path.exists():
        trained_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
    else:
        _train_noise_model(trained_model, train_loader, val_loader, config=config)
        trained_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))

    trained_model.eval()
    noise_abs_total = 0.0
    attribution_abs_total = 0.0
    noise_attention_shares = []
    with torch.no_grad():
        for batch in test_loader:
            (
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                noise_features,
                targets,
                user_ids,
                archetype_ids,
            ) = _collate_to_device(batch, device=device)
            _, _, noise_share = trained_model(
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                noise_features,
                user_ids=user_ids,
                archetype_ids=archetype_ids,
            )
            noise_attention_shares.append(float(noise_share.item()))

    for batch in test_loader:
        (
            hr_sequence,
            glucose_context,
            ecg_features,
            emg_features,
            eeg_signal,
            cbf_signal,
            noise_features,
            targets,
            user_ids,
            archetype_ids,
        ) = _collate_to_device(batch, device=device)
        noise_abs, total_abs = _noise_integrated_gradients(
            trained_model,
            (
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                noise_features,
                targets,
                user_ids,
                archetype_ids,
            ),
            device=device,
            n_steps=max(10, int(config["ig_n_steps"]) // 2),
        )
        noise_abs_total += noise_abs
        attribution_abs_total += total_abs

    control_model = NoiseAwareModel(base_model_path, config, zero_init_noise=True).to(device)
    control_model.base_model.set_known_user_ids(train_data.get_known_user_ids())
    control_model.eval()
    control_noise_shares = []
    with torch.no_grad():
        for batch in test_loader:
            (
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                noise_features,
                targets,
                user_ids,
                archetype_ids,
            ) = _collate_to_device(batch, device=device)
            _, _, noise_share = control_model(
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                noise_features,
                user_ids=user_ids,
                archetype_ids=archetype_ids,
            )
            control_noise_shares.append(float(noise_share.item()))

    payload = {
        "noise_ig_total_pct": (noise_abs_total / max(attribution_abs_total, 1e-6)) * 100.0,
        "noise_attention_share": float(np.mean(noise_attention_shares)) if noise_attention_shares else 0.0,
        "control_noise_attention_share": float(np.mean(control_noise_shares)) if control_noise_shares else 0.0,
        "n_train_windows": len(train_windows),
        "n_test_windows": len(test_windows),
        "checkpoint_path": str(checkpoint_path),
    }
    save_json(payload, result_path)
    return payload


__all__ = ["run_spurious_correlation_test"]
