"""Hierarchical EEG encoder for Part C."""

from __future__ import annotations

import torch
import torch.nn as nn

from part_a.model import AttentionTrackingEncoderLayer, SinusoidalPositionalEncoding
from part_c.models.common import TrackedSequenceEncoder, attention_rollout_profile


class LocalEEGEncoder(nn.Module):
    """Encode a 5-second EEG chunk into one local summary vector.

    Each 5-second window is split into 20 patches of 64 samples, embedded into a
    compact 32-dimensional space, processed by a single lightweight Transformer
    layer, and mean-pooled into one local summary token.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.patch_size = int(config["eeg_local_patch_size"])
        self.n_patches = int(config["eeg_local_patches"])
        self.d_model = int(config["local_d_model"])

        self.patch_projection = nn.Linear(self.patch_size, self.d_model)
        self.position_encoder = SinusoidalPositionalEncoding(
            d_model=self.d_model,
            max_seq_len=self.n_patches,
            dropout=config["dropout"],
        )
        encoder_layer = AttentionTrackingEncoderLayer(
            d_model=self.d_model,
            nhead=config["n_heads"],
            dim_feedforward=self.d_model * 4,
            dropout=config["dropout"],
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=1)

    def forward(self, eeg_window: torch.Tensor) -> torch.Tensor:
        """Return one summary vector per local EEG window."""

        patches = eeg_window.view(eeg_window.size(0), self.n_patches, self.patch_size)
        projected = self.patch_projection(patches)
        positioned = self.position_encoder(projected)
        encoded = self.encoder(positioned)
        return encoded.mean(dim=1)


class GlobalEEGEncoder(nn.Module):
    """Encode the sequence of local EEG summaries into one global representation."""

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = TrackedSequenceEncoder(
            n_features=int(config["local_d_model"]),
            d_model=config["d_model"],
            max_tokens=int(config["eeg_local_windows"]),
            n_heads=config["n_heads"],
            n_layers=config["n_encoder_layers_per_modal"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            use_modality_embedding=True,
            use_cls_token=True,
            checkpoint_layers=bool(config["eeg_gradient_checkpointing"]),
        )

    def forward(
        self,
        local_summaries: torch.Tensor,
        *,
        capture_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the global encoder over the sequence of local EEG summaries."""

        return self.encoder(local_summaries, capture_attention=capture_attention)


class HierarchicalEEGEncoder(nn.Module):
    """Two-stage hierarchical EEG encoder with exact local and global attention."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.local_windows = int(config["eeg_local_windows"])
        self.local_window_samples = int(config["eeg_local_window_samples"])
        self.local_encoder = LocalEEGEncoder(config)
        self.global_encoder = GlobalEEGEncoder(config)

    def _local_summaries(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Process local EEG windows sequentially to minimise peak memory."""

        signal = eeg_signal[:, : self.local_windows * self.local_window_samples]
        windows = signal.view(signal.size(0), self.local_windows, self.local_window_samples)

        summaries: list[torch.Tensor] = []
        for window_index in range(self.local_windows):
            local_summary = self.local_encoder(windows[:, window_index, :])
            summaries.append(local_summary)
        return torch.stack(summaries, dim=1)

    def forward(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Return the global CLS summary vector for the EEG sequence."""

        local_summaries = self._local_summaries(eeg_signal)
        encoded, _ = self.global_encoder(local_summaries, capture_attention=False)
        return encoded[:, 0, :]

    def get_attention_profile(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Return a 24-token attention profile over local EEG windows."""

        was_training = self.training
        self.eval()
        with torch.no_grad():
            local_summaries = self._local_summaries(eeg_signal)
            _, attention_weights = self.global_encoder(local_summaries, capture_attention=True)
            profile = attention_rollout_profile(attention_weights, has_cls_token=True)
        if was_training:
            self.train()
        return profile
