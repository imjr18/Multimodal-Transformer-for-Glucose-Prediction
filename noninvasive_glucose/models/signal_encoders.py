"""Standalone modality encoders for the non-invasive system."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Project low-dimensional biosignal features into the shared model space."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map `[batch, seq_len, n_features]` to `[batch, seq_len, d_model]`."""

        return self.projection(inputs)


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic order information to a parallel Transformer sequence."""

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        positional = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        positional[:, 0::2] = torch.sin(position * div_term)
        positional[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("positional_encoding", positional.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add the correct positional slice to each token and apply dropout."""

        sequence_length = inputs.size(1)
        positioned = inputs + self.positional_encoding[:, :sequence_length]
        return self.dropout(positioned)


class AttentionTrackingEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer that can retain attention maps for analysis."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_attention = False
        self.latest_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Run the layer while keeping the standard Transformer computation."""

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Apply self-attention and optionally retain per-head weights."""

        attended, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.capture_attention,
            average_attn_weights=False,
            is_causal=is_causal,
        )
        self.latest_attention_weights = weights.detach().cpu() if self.capture_attention and weights is not None else None
        return self.dropout1(attended)


class BaseSignalEncoder(nn.Module):
    """Reusable compact Transformer encoder for one biosignal modality.

    Each encoder keeps modality semantics separate during the first stage of
    processing. Token embeddings project raw features into `d_model`, a learned
    modality-type vector marks the token stream identity, sinusoidal positions
    encode order, and a small pre-layer-normalised encoder stack refines the
    sequence into contextual tokens of shape `[batch, seq_len, d_model]`.
    """

    def __init__(self, n_features: int, config: dict):
        super().__init__()
        self.config = config
        self.d_model = int(config["d_model"])
        self.token_embedding = TokenEmbedding(n_features=n_features, d_model=self.d_model)
        self.modality_embedding = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.position_encoder = SinusoidalPositionalEncoding(
            d_model=self.d_model,
            max_seq_len=int(config["max_seq_len"]),
            dropout=float(config["dropout"]),
        )
        layer = AttentionTrackingEncoderLayer(
            d_model=self.d_model,
            nhead=int(config["n_heads"]),
            dim_feedforward=int(config["d_ff"]),
            dropout=float(config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config["n_encoder_layers"]))

    def forward(self, inputs: torch.Tensor, *, capture_attention: bool = False) -> torch.Tensor:
        """Return contextualised modality tokens with shape `[batch, seq_len, d_model]`."""

        hidden = self.token_embedding(inputs) + self.modality_embedding
        hidden = self.position_encoder(hidden)
        for layer in self.encoder.layers:
            layer.capture_attention = capture_attention
            hidden = layer(hidden)
        if self.encoder.norm is not None:
            hidden = self.encoder.norm(hidden)
        return hidden

    def pooled(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mean-pool the encoded sequence for auxiliary pretraining tasks."""

        return self.forward(inputs).mean(dim=1)

    def get_attention_weights(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Return per-layer self-attention maps for interpretability."""

        was_training = self.training
        self.eval()
        with torch.no_grad():
            _ = self.forward(inputs, capture_attention=True)
            weights = [
                layer.latest_attention_weights
                for layer in self.encoder.layers
                if layer.latest_attention_weights is not None
            ]
        if was_training:
            self.train()
        return weights


class HREncoder(BaseSignalEncoder):
    """Encode heart-rate windows into contextual tokens."""

    def __init__(self, config: dict):
        super().__init__(n_features=1, config=config)


class ECGEncoder(BaseSignalEncoder):
    """Encode ECG-derived HRV features for later fusion."""

    def __init__(self, config: dict):
        super().__init__(n_features=5, config=config)


class EMGEncoder(BaseSignalEncoder):
    """Encode EMG envelope features for activity-state inference."""

    def __init__(self, config: dict):
        super().__init__(n_features=2, config=config)


class EEGBandEncoder(BaseSignalEncoder):
    """Encode 5-band EEG power features for sleep-state-aware inference."""

    def __init__(self, config: dict):
        super().__init__(n_features=5, config=config)


class CBFEncoder(BaseSignalEncoder):
    """Encode slow cerebral blood flow context."""

    def __init__(self, config: dict):
        super().__init__(n_features=1, config=config)


__all__ = [
    "AttentionTrackingEncoderLayer",
    "BaseSignalEncoder",
    "CBFEncoder",
    "ECGEncoder",
    "EEGBandEncoder",
    "EMGEncoder",
    "HREncoder",
    "SinusoidalPositionalEncoding",
    "TokenEmbedding",
]

