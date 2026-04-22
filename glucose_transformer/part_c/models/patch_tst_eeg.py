"""Patch-based EEG encoder for Part C."""

from __future__ import annotations

import torch
import torch.nn as nn

from part_c.models.common import TrackedSequenceEncoder, attention_rollout_profile


class PatchEEGEncoder(nn.Module):
    """Encode EEG by dividing it into non-overlapping raw waveform patches.

    A 2-minute EEG segment at 256 Hz becomes 480 patches of 64 samples each.
    Attention is then computed over the patch sequence instead of over the raw
    30,720-sample waveform. This is the PatchTST-style compromise between raw
    waveform access and tractable sequence length.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.patch_size = int(config["eeg_patch_size"])
        self.n_patches = int(config["eeg_patch_tokens"])

        self.encoder = TrackedSequenceEncoder(
            n_features=self.patch_size,
            d_model=config["d_model"],
            max_tokens=self.n_patches,
            n_heads=config["n_heads"],
            n_layers=config["n_encoder_layers_per_modal"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            use_modality_embedding=True,
            use_cls_token=False,
            checkpoint_layers=bool(config["eeg_gradient_checkpointing"]),
        )

    def _patchify(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Reshape raw EEG into `[batch, n_patches, patch_size]`."""

        signal = eeg_signal[:, : self.n_patches * self.patch_size]
        return signal.view(signal.size(0), self.n_patches, self.patch_size)

    def forward(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Return a mean-pooled patch-Transformer summary."""

        patches = self._patchify(eeg_signal)
        encoded, _ = self.encoder(patches, capture_attention=False)
        return encoded.mean(dim=1)

    def get_attention_profile(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Return a token-importance profile over the EEG patches."""

        was_training = self.training
        self.eval()
        with torch.no_grad():
            patches = self._patchify(eeg_signal)
            _, attention_weights = self.encoder(patches, capture_attention=True)
            profile = attention_rollout_profile(attention_weights, has_cls_token=False)
        if was_training:
            self.train()
        return profile
