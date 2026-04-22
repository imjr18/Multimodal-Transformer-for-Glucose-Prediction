"""Shared model helpers for Part B multimodal architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from part_a.model import (
    AttentionTrackingEncoderLayer,
    SinusoidalPositionalEncoding,
    TokenEmbedding,
)


def apply_modality_dropout(
    ecg_features: torch.Tensor,
    emg_features: torch.Tensor,
    p_drop: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly zero entire secondary modalities during training.

    The operation is applied at the batch level, independently for ECG and EMG,
    and never touches the primary HR stream. This matches the prompt's training
    design and makes ablation-style missing-modality inference less brittle.
    """

    if torch.rand(1, device=ecg_features.device) < p_drop:
        ecg_features = torch.zeros_like(ecg_features)
    if torch.rand(1, device=emg_features.device) < p_drop:
        emg_features = torch.zeros_like(emg_features)
    return ecg_features, emg_features


class SequenceEncoder(nn.Module):
    """Reusable Transformer encoder block with a learnable CLS token.

    This module is the Part B analogue of the Part A sequence backbone. It
    embeds a modality-specific feature stream, optionally adds a modality-type
    embedding, prepends a learnable CLS token, injects sinusoidal positions, and
    runs the sequence through a compact pre-layer-normalised encoder stack.
    """

    def __init__(
        self,
        n_features: int,
        config: dict,
        *,
        use_modality_embedding: bool = False,
    ):
        super().__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.use_gradient_checkpointing = bool(config.get("gradient_checkpointing", False))

        self.token_embedding = TokenEmbedding(n_features=n_features, d_model=self.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.modality_embedding = (
            nn.Parameter(torch.randn(1, 1, self.d_model))
            if use_modality_embedding
            else None
        )
        self.position_encoder = SinusoidalPositionalEncoding(
            d_model=self.d_model,
            max_seq_len=config["max_seq_len"],
            dropout=config["dropout"],
        )

        encoder_layer = AttentionTrackingEncoderLayer(
            d_model=self.d_model,
            nhead=config["n_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        num_layers = config.get("n_encoder_layers_per_modal", config["n_encoder_layers"])
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

    def _run_encoder(self, sequence: torch.Tensor) -> torch.Tensor:
        """Run the stacked encoder layers with optional checkpointing."""

        hidden = sequence
        for layer in self.encoder.layers:
            layer.capture_attention = False

        if self.use_gradient_checkpointing and self.training:
            for layer in self.encoder.layers:
                hidden = checkpoint(
                    lambda tensor, encoder_layer=layer: encoder_layer(tensor),
                    hidden,
                    use_reentrant=False,
                )
        else:
            for layer in self.encoder.layers:
                hidden = layer(hidden)

        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return hidden

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a modality sequence and return `[batch, seq_len+1, d_model]`."""

        batch_size = inputs.size(0)
        embedded = self.token_embedding(inputs)
        if self.modality_embedding is not None:
            embedded = embedded + self.modality_embedding

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, embedded], dim=1)
        positioned = self.position_encoder(sequence)
        return self._run_encoder(positioned)


def build_regression_head(config: dict) -> nn.Sequential:
    """Create the standard two-layer regression head used across Part B models."""

    return nn.Sequential(
        nn.Linear(config["d_model"], 64),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(64, len(config["target_offsets"])),
    )
