"""Cross-attention fusion adapted for the non-invasive task."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Fuse multimodal biosignal tokens through HR-centric cross-attention.

    Heart rate is used as the primary query stream because autonomic state is a
    broad physiological hub. ECG-HRV, EMG, EEG band powers, and CBF each supply
    additional context through separate cross-attention layers. Their outputs
    are added back to the HR representation in residual style and normalised.
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        dropout = float(config["dropout"])
        self.hr_to_ecg = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.hr_to_emg = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.hr_to_eeg = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.hr_to_cbf = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.latest_attention: dict[str, torch.Tensor] = {}

    def forward(
        self,
        hr_encoded: torch.Tensor,
        ecg_encoded: torch.Tensor,
        emg_encoded: torch.Tensor,
        eeg_encoded: torch.Tensor,
        cbf_encoded: torch.Tensor,
        *,
        capture_attention: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a fused HR sequence and optionally the cross-attention maps."""

        hr_ecg, hr_ecg_w = self.hr_to_ecg(
            query=hr_encoded,
            key=ecg_encoded,
            value=ecg_encoded,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        hr_emg, hr_emg_w = self.hr_to_emg(
            query=hr_encoded,
            key=emg_encoded,
            value=emg_encoded,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        hr_eeg, hr_eeg_w = self.hr_to_eeg(
            query=hr_encoded,
            key=eeg_encoded,
            value=eeg_encoded,
            need_weights=capture_attention,
            average_attn_weights=False,
        )
        hr_cbf, hr_cbf_w = self.hr_to_cbf(
            query=hr_encoded,
            key=cbf_encoded,
            value=cbf_encoded,
            need_weights=capture_attention,
            average_attn_weights=False,
        )

        fused = self.fusion_norm(hr_encoded + hr_ecg + hr_emg + hr_eeg + hr_cbf)
        attention_payload: dict[str, torch.Tensor] = {}
        if capture_attention:
            attention_payload = {
                "hr_to_ecg": hr_ecg_w.detach().cpu() if hr_ecg_w is not None else None,
                "hr_to_emg": hr_emg_w.detach().cpu() if hr_emg_w is not None else None,
                "hr_to_eeg": hr_eeg_w.detach().cpu() if hr_eeg_w is not None else None,
                "hr_to_cbf": hr_cbf_w.detach().cpu() if hr_cbf_w is not None else None,
            }
            self.latest_attention = attention_payload
        return fused, attention_payload


__all__ = ["CrossAttentionFusion"]

