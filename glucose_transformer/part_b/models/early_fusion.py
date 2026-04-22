"""Early-fusion Transformer for Part B."""

from __future__ import annotations

import torch
import torch.nn as nn

from part_b.models.common import SequenceEncoder, apply_modality_dropout, build_regression_head


class EarlyFusionTransformer(nn.Module):
    """Concatenate all modalities before any temporal encoding.

    The model forms a 9-feature token at each timestep:
    HR (1) + glucose context (1) + ECG-HRV (5) + EMG (2). A single encoder then
    learns temporal structure over that fused stream. This is the cheapest Part
    B baseline, but it gives up modality-specific processing before the model has
    a chance to understand each signal on its own.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.modality_dropout_p = float(config["modality_dropout_p"])
        self.encoder = SequenceEncoder(
            n_features=config["early_fusion_feature_dim"],
            config=config,
            use_modality_embedding=False,
        )
        self.regression_head = build_regression_head(config)

    def forward(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
    ) -> torch.Tensor:
        """Encode an early-fused multimodal sequence and predict two horizons."""

        if self.training:
            ecg_features, emg_features = apply_modality_dropout(
                ecg_features,
                emg_features,
                p_drop=self.modality_dropout_p,
            )

        fused_inputs = torch.cat([hr_seq, glucose_context, ecg_features, emg_features], dim=-1)
        encoded_sequence = self.encoder(fused_inputs)
        cls_representation = encoded_sequence[:, 0, :]
        return self.regression_head(cls_representation)
