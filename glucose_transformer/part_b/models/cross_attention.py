"""Cross-attention multimodal Transformer for Part B."""

from __future__ import annotations

import torch
import torch.nn as nn

from part_b.models.common import SequenceEncoder, apply_modality_dropout, build_regression_head


class ModalityEncoder(SequenceEncoder):
    """Single-modality encoder with modality-type embedding support.

    This encoder reuses the Part A token embedding and positional encoding
    machinery while adding a learned modality identifier to every token before
    self-attention. The output keeps a leading CLS token so later fusion blocks
    can produce a sequence summary in the same way across modalities.
    """

    def __init__(self, n_features: int, config: dict):
        super().__init__(n_features=n_features, config=config, use_modality_embedding=True)


class CrossModalTransformer(nn.Module):
    """Three modality encoders with HR-centric cross-attention fusion.

    HR plus glucose context is treated as the primary sequence. ECG-HRV and EMG
    are first encoded independently, then HR queries each of them through
    separate multi-head cross-attention layers. The cross-modal outputs are added
    back to the HR representation in residual style, normalised, and decoded
    through the HR CLS token.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.modality_dropout_p = float(config["modality_dropout_p"])
        self.capture_cross_attention = False
        self.latest_cross_attention_weights: dict[str, torch.Tensor] = {}

        self.hr_encoder = ModalityEncoder(n_features=config["hr_feature_dim"], config=config)
        self.ecg_encoder = ModalityEncoder(n_features=config["ecg_feature_dim"], config=config)
        self.emg_encoder = ModalityEncoder(n_features=config["emg_feature_dim"], config=config)

        self.hr_to_ecg_attention = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.hr_to_emg_attention = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(config["d_model"])
        self.regression_head = build_regression_head(config)

    def _clear_cache_if_needed(self, reference_tensor: torch.Tensor) -> None:
        """Release cached CUDA memory when running on GPU."""

        if torch.cuda.is_available() and reference_tensor.is_cuda:
            torch.cuda.empty_cache()

    def _forward_impl(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        *,
        capture_attention: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Shared implementation for regular inference and attention extraction."""

        if self.training and not capture_attention:
            ecg_features, emg_features = apply_modality_dropout(
                ecg_features,
                emg_features,
                p_drop=self.modality_dropout_p,
            )

        hr_inputs = torch.cat([hr_seq, glucose_context], dim=-1)

        hr_encoded = self.hr_encoder(hr_inputs)
        ecg_encoded = self.ecg_encoder(ecg_features)

        hr_enriched_with_ecg, hr_to_ecg_weights = self.hr_to_ecg_attention(
            query=hr_encoded,
            key=ecg_encoded,
            value=ecg_encoded,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        del ecg_encoded
        self._clear_cache_if_needed(hr_encoded)

        emg_encoded = self.emg_encoder(emg_features)
        hr_enriched_with_emg, hr_to_emg_weights = self.hr_to_emg_attention(
            query=hr_encoded,
            key=emg_encoded,
            value=emg_encoded,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        del emg_encoded
        self._clear_cache_if_needed(hr_encoded)

        hr_fused = self.fusion_norm(hr_encoded + hr_enriched_with_ecg + hr_enriched_with_emg)
        predictions = self.regression_head(hr_fused[:, 0, :])

        attention_payload: dict[str, torch.Tensor] = {}
        if capture_attention and hr_to_ecg_weights is not None and hr_to_emg_weights is not None:
            attention_payload = {
                "hr_to_ecg": hr_to_ecg_weights.detach().cpu()[:, :, 1:, 1:],
                "hr_to_emg": hr_to_emg_weights.detach().cpu()[:, :, 1:, 1:],
            }
        return predictions, attention_payload

    def forward(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future glucose from HR, glucose context, ECG-HRV, and EMG."""

        predictions, _ = self._forward_impl(
            hr_seq,
            glucose_context,
            ecg_features,
            emg_features,
            capture_attention=False,
        )
        return predictions

    def get_cross_attention_weights(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return HR-to-ECG and HR-to-EMG attention maps without the CLS token.

        The returned tensors have shape `[batch, n_heads, seq_len, seq_len]`,
        which matches the Part B visualisation and interpretability requirement.
        """

        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, attention_payload = self._forward_impl(
                hr_seq,
                glucose_context,
                ecg_features,
                emg_features,
                capture_attention=True,
            )
        if was_training:
            self.train()
        self.latest_cross_attention_weights = attention_payload
        return attention_payload
