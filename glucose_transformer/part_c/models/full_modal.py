"""Full multimodal Transformer backbone for Part C."""

from __future__ import annotations

import torch
import torch.nn as nn

from part_a.model import AttentionTrackingEncoderLayer
from part_b.models.common import build_regression_head
from part_b.models.cross_attention import CrossModalTransformer, ModalityEncoder
from part_c.models.frequency_eeg import FrequencyEEGEncoder
from part_c.models.hierarchical_eeg import HierarchicalEEGEncoder
from part_c.models.patch_tst_eeg import PatchEEGEncoder


def build_eeg_encoder(eeg_encoder_kind: str, config: dict) -> nn.Module:
    """Factory for the three Part C EEG encoder variants."""

    if eeg_encoder_kind == "frequency_eeg":
        return FrequencyEEGEncoder(config)
    if eeg_encoder_kind == "patch_eeg":
        return PatchEEGEncoder(config)
    if eeg_encoder_kind == "hierarchical_eeg":
        return HierarchicalEEGEncoder(config)
    raise ValueError(f"Unsupported EEG encoder kind: {eeg_encoder_kind}")


class FullModalTransformer(CrossModalTransformer):
    """Extend the Part B HR-centric cross-modal backbone with EEG and CBF.

    The model keeps HR, ECG-HRV, and EMG sequence fusion from Part B, adds a
    CBF summary through an additional HR-to-CBF cross-attention block, and
    appends an EEG summary token before a final lightweight fusion encoder. EEG
    is processed first so the large raw waveform can be discarded before the
    rest of the multimodal stack is materialised.
    """

    def __init__(self, config: dict, *, eeg_encoder_kind: str):
        super().__init__(config)
        self.config = config
        self.eeg_encoder_kind = eeg_encoder_kind
        self.eeg_encoder = build_eeg_encoder(eeg_encoder_kind, config)
        self.cbf_encoder = ModalityEncoder(n_features=config["cbf_feature_dim"], config=config)
        self.hr_to_cbf_attention = nn.MultiheadAttention(
            embed_dim=config["d_model"],
            num_heads=config["n_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.eeg_summary_projection = nn.Linear(config["d_model"], config["d_model"])
        final_layer = AttentionTrackingEncoderLayer(
            d_model=config["d_model"],
            nhead=config["n_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.final_fusion_encoder = nn.TransformerEncoder(final_layer, num_layers=1)
        self.regression_head = build_regression_head(config)

    def _forward_impl(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        eeg_signal: torch.Tensor,
        cbf_signal: torch.Tensor,
        *,
        capture_attention: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Shared implementation for inference and attention extraction."""

        eeg_summary = self.eeg_encoder(eeg_signal)
        del eeg_signal
        self._clear_cache_if_needed(eeg_summary)

        if self.training and not capture_attention:
            ecg_features, emg_features = self.apply_modality_dropout(ecg_features, emg_features)

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

        cbf_encoded = self.cbf_encoder(cbf_signal)
        cbf_summary = cbf_encoded[:, 0, :]
        del cbf_encoded
        cbf_context = cbf_summary.unsqueeze(1).expand(-1, hr_encoded.size(1), -1)
        hr_enriched_with_cbf, hr_to_cbf_weights = self.hr_to_cbf_attention(
            query=hr_encoded,
            key=cbf_context,
            value=cbf_context,
            need_weights=capture_attention,
            average_attn_weights=False,
        )

        hr_fused = self.fusion_norm(hr_encoded + hr_enriched_with_ecg + hr_enriched_with_emg + hr_enriched_with_cbf)
        eeg_token = self.eeg_summary_projection(eeg_summary).unsqueeze(1)
        fused_sequence = torch.cat([hr_fused, eeg_token], dim=1)
        fused_encoded = self.final_fusion_encoder(fused_sequence)
        predictions = self.regression_head(fused_encoded[:, 0, :])

        attention_payload: dict[str, torch.Tensor] = {}
        if capture_attention:
            attention_payload = {
                "hr_to_ecg": hr_to_ecg_weights.detach().cpu()[:, :, 1:, 1:]
                if hr_to_ecg_weights is not None
                else None,
                "hr_to_emg": hr_to_emg_weights.detach().cpu()[:, :, 1:, 1:]
                if hr_to_emg_weights is not None
                else None,
                "hr_to_cbf": hr_to_cbf_weights.detach().cpu()[:, :, 1:, 1:]
                if hr_to_cbf_weights is not None
                else None,
            }
        return predictions, attention_payload

    def apply_modality_dropout(
        self,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Proxy the Part B modality-dropout helper."""

        from part_b.models.common import apply_modality_dropout

        return apply_modality_dropout(
            ecg_features,
            emg_features,
            p_drop=self.modality_dropout_p,
        )

    def forward(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        eeg_signal: torch.Tensor,
        cbf_signal: torch.Tensor,
    ) -> torch.Tensor:
        """Predict future glucose from the full Part C modality set."""

        predictions, _ = self._forward_impl(
            hr_seq,
            glucose_context,
            ecg_features,
            emg_features,
            eeg_signal,
            cbf_signal,
            capture_attention=False,
        )
        return predictions

    def get_cross_attention_weights(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        eeg_signal: torch.Tensor,
        cbf_signal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return HR cross-attention weights for ECG, EMG, and CBF."""

        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, attention_payload = self._forward_impl(
                hr_seq,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
                capture_attention=True,
            )
        if was_training:
            self.train()
        return attention_payload

    def get_eeg_attention_profile(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Delegate attention-profile extraction to the selected EEG encoder."""

        return self.eeg_encoder.get_attention_profile(eeg_signal)
