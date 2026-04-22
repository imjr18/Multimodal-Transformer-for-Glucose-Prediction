"""Transformer model definitions for Part A glucose forecasting."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TokenEmbedding(nn.Module):
    """Project raw scalar biosignal values into the transformer's model space.

    Transformers operate on vectors, not scalar measurements. This layer turns a
    one-feature input such as heart rate or glucose context into a dense
    `d_model` representation that can later participate in self-attention.
    """

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map `[batch, seq_len, n_features]` tensors to `[batch, seq_len, d_model]`."""

        return self.projection(inputs)


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic position information to token embeddings.

    Self-attention processes all timesteps in parallel, so without an explicit
    positional signal the model would know the values but not their order. The
    sinusoidal encoding gives every position a unique pattern across dimensions,
    allowing the model to reason about temporal distance and sequence order.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        positional_encoding = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("positional_encoding", positional_encoding.unsqueeze(0), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Add position vectors to token embeddings and apply dropout."""

        sequence_length = inputs.size(1)
        # Every timestep gets a unique encoding because the transformer sees the
        # whole sequence at once and otherwise cannot infer which token happened when.
        positioned = inputs + self.positional_encoding[:, :sequence_length]
        return self.dropout(positioned)


class AttentionTrackingEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer that can retain per-head attention weights.

    PyTorch's stock encoder layer does not keep attention maps because
    `need_weights=False` is the efficient default. This subclass keeps the same
    architecture and training behaviour while optionally exposing the weights for
    post-hoc visualisation.
    """

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
        """Run the encoder layer while bypassing the fast path so weights can be stored."""

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Run self-attention and optionally keep the per-head weight tensor."""

        attended, attention_weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=self.capture_attention,
            average_attn_weights=False,
            is_causal=is_causal,
        )

        if self.capture_attention and attention_weights is not None:
            self.latest_attention_weights = attention_weights.detach().cpu()
        else:
            self.latest_attention_weights = None

        return self.dropout1(attended)


class TemporalTransformer(nn.Module):
    """Temporal Transformer encoder for 30- and 60-minute glucose forecasting.

    The model embeds heart-rate and glucose-context streams separately, fuses
    them into a shared `d_model` representation, prepends a learnable CLS token,
    injects sinusoidal positions, and passes the sequence through a small
    pre-layer-normalised Transformer encoder. The final CLS representation is
    decoded by a regression head into two glucose forecasts.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config
        self.d_model = config["d_model"]
        self.use_gradient_checkpointing = bool(config.get("gradient_checkpointing", False))

        self.hr_embedding = TokenEmbedding(n_features=1, d_model=self.d_model)
        self.glucose_embedding = TokenEmbedding(n_features=1, d_model=self.d_model)
        self.fusion_projection = nn.Linear(self.d_model * 2, self.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
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
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config["n_encoder_layers"],
        )

        self.regression_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(64, len(config["target_offsets"])),
        )

    def _run_encoder(self, sequence: torch.Tensor, *, capture_attention: bool) -> torch.Tensor:
        """Run the encoder stack with optional checkpointing and attention capture."""

        hidden = sequence
        for layer in self.encoder.layers:
            layer.capture_attention = capture_attention

        if self.use_gradient_checkpointing and self.training and not capture_attention:
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

    def _forward_impl(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
        *,
        capture_attention: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Shared forward implementation for prediction and attention extraction."""

        batch_size = hr_seq.size(0)

        # hr_seq: [batch, seq_len=24, features=1]
        hr_tokens = self.hr_embedding(hr_seq)
        # after HR embedding: [batch, seq_len=24, d_model=64]

        # glucose_context: [batch, seq_len=24, features=1]
        glucose_tokens = self.glucose_embedding(glucose_context)
        # after glucose embedding: [batch, seq_len=24, d_model=64]

        fused_tokens = torch.cat([hr_tokens, glucose_tokens], dim=-1)
        # after concatenation: [batch, seq_len=24, d_model*2=128]

        fused_tokens = self.fusion_projection(fused_tokens)
        # after fusion projection: [batch, seq_len=24, d_model=64]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # cls_tokens: [batch, 1, d_model=64]

        sequence = torch.cat([cls_tokens, fused_tokens], dim=1)
        # sequence with CLS: [batch, seq_len+1=25, d_model=64]

        positioned_sequence = self.position_encoder(sequence)
        # after positional encoding: [batch, seq_len+1=25, d_model=64]

        encoded_sequence = self._run_encoder(positioned_sequence, capture_attention=capture_attention)
        # after encoder: [batch, seq_len+1=25, d_model=64]

        cls_representation = encoded_sequence[:, 0, :]
        # CLS slice: [batch, d_model=64]

        predictions = self.regression_head(cls_representation)
        # regression output: [batch, 2]

        attention_weights = [
            layer.latest_attention_weights
            for layer in self.encoder.layers
            if layer.latest_attention_weights is not None
        ]
        return predictions, attention_weights

    def forward(self, hr_seq: torch.Tensor, glucose_context: torch.Tensor) -> torch.Tensor:
        """Predict normalised glucose values at +30 and +60 minutes."""

        predictions, _ = self._forward_impl(hr_seq, glucose_context, capture_attention=False)
        return predictions

    def get_attention_weights(
        self,
        hr_seq: torch.Tensor,
        glucose_context: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Return per-layer, per-head attention maps for visualisation.

        Each tensor has shape `[batch, n_heads, seq_len + 1, seq_len + 1]`,
        where the extra position corresponds to the prepended CLS token.
        """

        was_training = self.training
        self.eval()
        with torch.no_grad():
            _, attention_weights = self._forward_impl(
                hr_seq,
                glucose_context,
                capture_attention=True,
            )
        if was_training:
            self.train()
        return attention_weights
