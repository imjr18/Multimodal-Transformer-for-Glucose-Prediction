"""Shared model utilities for Part C efficient EEG encoders."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from part_a.model import AttentionTrackingEncoderLayer, SinusoidalPositionalEncoding, TokenEmbedding


class TrackedSequenceEncoder(nn.Module):
    """Transformer encoder with optional modality embeddings, CLS token, and attention capture."""

    def __init__(
        self,
        *,
        n_features: int,
        d_model: int,
        max_tokens: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        use_modality_embedding: bool = False,
        use_cls_token: bool = True,
        checkpoint_layers: bool = False,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.checkpoint_layers = checkpoint_layers
        self.token_embedding = TokenEmbedding(n_features=n_features, d_model=d_model)
        self.modality_embedding = (
            nn.Parameter(torch.randn(1, 1, d_model)) if use_modality_embedding else None
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if use_cls_token else None

        max_seq_len = max_tokens + (1 if use_cls_token else 0)
        self.position_encoder = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        encoder_layer = AttentionTrackingEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        capture_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode a sequence and optionally retain attention weights."""

        batch_size = inputs.size(0)
        hidden = self.token_embedding(inputs)
        if self.modality_embedding is not None:
            hidden = hidden + self.modality_embedding
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden = torch.cat([cls_tokens, hidden], dim=1)

        hidden = self.position_encoder(hidden)

        for layer in self.encoder.layers:
            layer.capture_attention = capture_attention

        if self.checkpoint_layers and self.training and not capture_attention:
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

        attention_weights = [
            layer.latest_attention_weights
            for layer in self.encoder.layers
            if layer.latest_attention_weights is not None
        ]
        return hidden, attention_weights


def attention_rollout_profile(
    attention_weights: list[torch.Tensor],
    *,
    has_cls_token: bool,
) -> torch.Tensor:
    """Collapse per-layer attention maps into a single token-importance profile."""

    if not attention_weights:
        raise ValueError("No attention weights were captured.")

    batch_size = attention_weights[0].shape[0]
    sequence_length = attention_weights[0].shape[-1]
    rollout = torch.eye(sequence_length, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)

    for layer_weights in attention_weights:
        averaged_heads = layer_weights.detach().cpu().mean(dim=1)
        augmented_attention = averaged_heads + torch.eye(sequence_length, dtype=torch.float32).unsqueeze(0)
        augmented_attention = augmented_attention / augmented_attention.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(augmented_attention, rollout)

    if has_cls_token:
        return rollout[:, 0, 1:]

    return rollout.mean(dim=1)


def resample_profile(profile: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample a 1D attention profile to a fixed target length."""

    source = np.asarray(profile, dtype=np.float32).reshape(-1)
    if source.size == target_length:
        return source
    x_old = np.linspace(0.0, 1.0, source.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    return np.interp(x_new, x_old, source).astype("float32")
