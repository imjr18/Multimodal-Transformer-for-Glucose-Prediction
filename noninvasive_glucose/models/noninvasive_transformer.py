"""Full standalone multimodal Transformer for non-invasive glucose estimation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from noninvasive_glucose.models.fusion import CrossAttentionFusion
from noninvasive_glucose.models.signal_encoders import (
    AttentionTrackingEncoderLayer,
    CBFEncoder,
    ECGEncoder,
    EEGBandEncoder,
    EMGEncoder,
    HREncoder,
    SinusoidalPositionalEncoding,
)
from noninvasive_glucose.models.uncertainty_head import UncertaintyHead


class UserEmbeddingBank(nn.Module):
    """Lookup table for known users plus archetype prototypes for fallbacks."""

    def __init__(self, n_users: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.archetype_embedding = nn.Embedding(4, embedding_dim)
        self.register_buffer("known_user_mask", torch.zeros(n_users, dtype=torch.bool))

    def set_known_user_ids(self, user_ids: list[int]) -> None:
        """Mark which user IDs may use dedicated lookup embeddings."""

        mask = torch.zeros_like(self.known_user_mask)
        if user_ids:
            indices = torch.as_tensor(user_ids, dtype=torch.long)
            mask[indices] = True
        self.known_user_mask.copy_(mask)

    def resolve(
        self,
        *,
        batch_size: int,
        user_ids: torch.Tensor | None,
        archetype_ids: torch.Tensor | None,
        override: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Resolve one user-conditioning vector per batch element."""

        if override is not None:
            if override.dim() == 1:
                override = override.unsqueeze(0)
            if override.size(0) == 1 and batch_size > 1:
                override = override.expand(batch_size, -1)
            return override.to(device=device)

        if user_ids is not None:
            user_ids = user_ids.to(device=self.user_embedding.weight.device, dtype=torch.long)
            safe_ids = user_ids.clamp(min=0, max=self.user_embedding.num_embeddings - 1)
            embeddings = self.user_embedding(safe_ids)
            known_mask = self.known_user_mask[safe_ids]
        else:
            embeddings = None
            known_mask = None

        if archetype_ids is None:
            archetype_ids = torch.zeros(batch_size, dtype=torch.long, device=self.archetype_embedding.weight.device)
        else:
            archetype_ids = archetype_ids.to(device=self.archetype_embedding.weight.device, dtype=torch.long)
            if archetype_ids.dim() == 0:
                archetype_ids = archetype_ids.unsqueeze(0)
            if archetype_ids.size(0) == 1 and batch_size > 1:
                archetype_ids = archetype_ids.expand(batch_size)
        archetype_embeddings = self.archetype_embedding(archetype_ids)

        if embeddings is None or known_mask is None:
            resolved = archetype_embeddings
        else:
            resolved = torch.where(known_mask.unsqueeze(-1), embeddings, archetype_embeddings)
        return resolved.to(device=device)


class NonInvasiveTransformer(nn.Module):
    """Full non-invasive glucose estimation model.

    The model removes glucose context entirely, keeps one encoder per modality,
    fuses modalities through HR-centric cross-attention, adds a user token for
    personalisation, and decodes the final CLS token through a probabilistic
    uncertainty head.
    """

    def __init__(self, config: dict, *, n_users: int):
        super().__init__()
        self.config = config
        d_model = int(config["d_model"])

        self.hr_encoder = HREncoder(config)
        self.ecg_encoder = ECGEncoder(config)
        self.emg_encoder = EMGEncoder(config)
        self.eeg_encoder = EEGBandEncoder(config)
        self.cbf_encoder = CBFEncoder(config)
        self.user_embeddings = UserEmbeddingBank(n_users=n_users, embedding_dim=int(config["user_emb_dim"]))
        self.user_projection = nn.Linear(int(config["user_emb_dim"]), d_model)
        self.fusion = CrossAttentionFusion(config)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.final_position_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=int(config["max_seq_len"]),
            dropout=float(config["dropout"]),
        )
        final_layer = AttentionTrackingEncoderLayer(
            d_model=d_model,
            nhead=int(config["n_heads"]),
            dim_feedforward=int(config["d_ff"]),
            dropout=float(config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.final_encoder = nn.TransformerEncoder(final_layer, num_layers=1)
        self.uncertainty_head = UncertaintyHead(d_model=d_model, dropout=float(config["dropout"]))
        self.user_embedding_override: torch.Tensor | None = None
        self.latest_cross_attention: dict[str, torch.Tensor] = {}

    def set_calibration_embedding(self, embedding: torch.Tensor | None) -> None:
        """Store a calibrated embedding override for subsequent inference."""

        self.user_embedding_override = embedding.detach().clone() if embedding is not None else None

    def clear_calibration_embedding(self) -> None:
        """Remove any stored calibration override."""

        self.user_embedding_override = None

    def _final_encode(self, sequence: torch.Tensor, *, capture_attention: bool) -> torch.Tensor:
        """Run the last Transformer layer over `[CLS, fused_tokens, user_token]`."""

        hidden = self.final_position_encoder(sequence)
        for layer in self.final_encoder.layers:
            layer.capture_attention = capture_attention
            hidden = layer(hidden)
        if self.final_encoder.norm is not None:
            hidden = self.final_encoder.norm(hidden)
        return hidden

    def forward(
        self,
        hr: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        eeg_bands: torch.Tensor,
        cbf: torch.Tensor,
        *,
        user_ids: torch.Tensor | None = None,
        archetype_ids: torch.Tensor | None = None,
        user_embedding_override: torch.Tensor | None = None,
        capture_attention: bool = False,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Estimate current glucose mean and log-variance from biosignals only."""

        batch_size = hr.size(0)
        hr_encoded = self.hr_encoder(hr)
        ecg_encoded = self.ecg_encoder(ecg_features)
        emg_encoded = self.emg_encoder(emg_features)
        eeg_encoded = self.eeg_encoder(eeg_bands)
        cbf_encoded = self.cbf_encoder(cbf)

        fused_hr, attention_payload = self.fusion(
            hr_encoded,
            ecg_encoded,
            emg_encoded,
            eeg_encoded,
            cbf_encoded,
            capture_attention=capture_attention,
        )
        if capture_attention:
            self.latest_cross_attention = attention_payload

        resolved_user_embedding = self.user_embeddings.resolve(
            batch_size=batch_size,
            user_ids=user_ids,
            archetype_ids=archetype_ids,
            override=user_embedding_override if user_embedding_override is not None else self.user_embedding_override,
            device=hr.device,
        )
        user_token = self.user_projection(resolved_user_embedding).unsqueeze(1)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        final_sequence = torch.cat([cls_token, fused_hr, user_token], dim=1)
        encoded = self._final_encode(final_sequence, capture_attention=capture_attention)

        cls_representation = encoded[:, 0, :]
        mean, log_var = self.uncertainty_head(cls_representation)

        if return_aux:
            aux = {
                "hr_encoded": hr_encoded,
                "ecg_encoded": ecg_encoded,
                "emg_encoded": emg_encoded,
                "eeg_encoded": eeg_encoded,
                "cbf_encoded": cbf_encoded,
                "fused_hr": fused_hr,
                "final_sequence": encoded,
                "cross_attention": attention_payload,
            }
            return mean, log_var, aux
        return mean, log_var

    def predict_with_uncertainty(
        self,
        hr: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        eeg_bands: torch.Tensor,
        cbf: torch.Tensor,
        *,
        user_ids: torch.Tensor | None = None,
        archetype_ids: torch.Tensor | None = None,
        n_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run Monte Carlo Dropout inference and combine epistemic and aleatoric uncertainty."""

        n_draws = int(self.config["mc_dropout_samples"] if n_samples is None else n_samples)
        was_training = self.training
        self.train()
        use_amp = (
            bool(self.config.get("use_amp_inference", False))
            and torch.cuda.is_available()
            and hr.is_cuda
        )

        mean_samples: list[torch.Tensor] = []
        log_var_samples: list[torch.Tensor] = []
        with torch.no_grad():
            for _ in range(n_draws):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    mean, log_var = self(
                        hr,
                        ecg_features,
                        emg_features,
                        eeg_bands,
                        cbf,
                        user_ids=user_ids,
                        archetype_ids=archetype_ids,
                    )
                mean_samples.append(mean)
                log_var_samples.append(log_var)

        if not was_training:
            self.eval()

        mean_stack = torch.stack(mean_samples, dim=0)
        log_var_stack = torch.stack(log_var_samples, dim=0)
        predictive_mean = mean_stack.mean(dim=0)
        epistemic_var = mean_stack.var(dim=0, unbiased=False)
        aleatoric_var = torch.exp(log_var_stack).mean(dim=0)
        total_std = torch.sqrt(epistemic_var + aleatoric_var)

        return {
            "mean": predictive_mean,
            "epistemic_std": torch.sqrt(epistemic_var),
            "aleatoric_std": torch.sqrt(aleatoric_var),
            "total_std": total_std,
            "samples": mean_stack,
        }


__all__ = ["NonInvasiveTransformer", "UserEmbeddingBank"]
