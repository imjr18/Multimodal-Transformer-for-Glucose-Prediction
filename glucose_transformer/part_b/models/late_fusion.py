"""Late-fusion Transformer for Part B."""

from __future__ import annotations

import torch
import torch.nn as nn

from part_b.models.common import SequenceEncoder, apply_modality_dropout


class LateFusionTransformer(nn.Module):
    """Encode each modality independently and fuse only their summary vectors.

    HR plus glucose context, ECG-HRV, and EMG each get their own Transformer
    encoder. Their CLS summaries are concatenated into a single 192-dimensional
    vector that is reduced by a small fusion head. This keeps modality-specific
    representations clean but prevents token-level cross-modal interaction.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.modality_dropout_p = float(config["modality_dropout_p"])

        self.hr_encoder = SequenceEncoder(n_features=config["hr_feature_dim"], config=config)
        self.ecg_encoder = SequenceEncoder(n_features=config["ecg_feature_dim"], config=config)
        self.emg_encoder = SequenceEncoder(n_features=config["emg_feature_dim"], config=config)

        self.fusion_head = nn.Sequential(
            nn.Linear(config["d_model"] * 3, 64),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(64, len(config["target_offsets"])),
        )

    def forward(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode each modality separately, then fuse the three CLS summaries."""

        if self.training:
            ecg_features, emg_features = apply_modality_dropout(
                ecg_features,
                emg_features,
                p_drop=self.modality_dropout_p,
            )

        hr_inputs = torch.cat([hr_seq, glucose_context], dim=-1)
        hr_summary = self.hr_encoder(hr_inputs)[:, 0, :]
        if torch.cuda.is_available() and hr_summary.is_cuda:
            torch.cuda.empty_cache()

        ecg_summary = self.ecg_encoder(ecg_features)[:, 0, :]
        if torch.cuda.is_available() and ecg_summary.is_cuda:
            torch.cuda.empty_cache()

        emg_summary = self.emg_encoder(emg_features)[:, 0, :]

        fused_summary = torch.cat([hr_summary, ecg_summary, emg_summary], dim=-1)
        return self.fusion_head(fused_summary)
