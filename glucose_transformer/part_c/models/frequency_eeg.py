"""Frequency-feature EEG encoder for Part C."""

from __future__ import annotations

import torch
import torch.nn as nn

from part_c.models.common import TrackedSequenceEncoder, attention_rollout_profile


class FrequencyEEGEncoder(nn.Module):
    """Encode EEG as a sequence of 1-second relative band-power tokens.

    The raw 2-minute EEG is segmented into 120 non-overlapping 1-second windows.
    Each window is converted into five relative band powers, producing a short,
    interpretable sequence that a standard Transformer can process cheaply.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.sfreq = int(config["eeg_sfreq"])
        self.window_seconds = int(config["eeg_band_window_seconds"])
        self.samples_per_window = self.sfreq * self.window_seconds
        self.n_tokens = int(config["eeg_band_tokens"])

        self.encoder = TrackedSequenceEncoder(
            n_features=5,
            d_model=config["d_model"],
            max_tokens=self.n_tokens,
            n_heads=config["n_heads"],
            n_layers=config["n_encoder_layers_per_modal"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            use_modality_embedding=True,
            use_cls_token=False,
            checkpoint_layers=bool(config["eeg_gradient_checkpointing"]),
        )

    def _band_power_tokens(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Convert raw EEG into per-second relative band-power tokens."""

        batch_size = eeg_signal.size(0)
        signal = eeg_signal[:, : self.n_tokens * self.samples_per_window]
        segments = signal.view(batch_size, self.n_tokens, self.samples_per_window)

        fft = torch.fft.rfft(segments, dim=-1)
        power = fft.abs().pow(2)
        frequencies = torch.fft.rfftfreq(self.samples_per_window, d=1.0 / self.sfreq).to(signal.device)

        band_masks = [
            (frequencies >= 0.5) & (frequencies < 4.0),
            (frequencies >= 4.0) & (frequencies < 8.0),
            (frequencies >= 8.0) & (frequencies < 13.0),
            (frequencies >= 13.0) & (frequencies < 30.0),
            (frequencies >= 30.0) & (frequencies < 50.0),
        ]

        band_powers = []
        for mask in band_masks:
            band_power = power[..., mask].mean(dim=-1)
            band_powers.append(band_power)

        tokens = torch.stack(band_powers, dim=-1)
        total_power = tokens.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return tokens / total_power

    def forward(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Return a single EEG summary vector of shape `[batch, d_model]`."""

        band_tokens = self._band_power_tokens(eeg_signal)
        encoded, _ = self.encoder(band_tokens, capture_attention=False)
        return encoded.mean(dim=1)

    def get_attention_profile(self, eeg_signal: torch.Tensor) -> torch.Tensor:
        """Return a token-importance profile over the 120 EEG seconds."""

        was_training = self.training
        self.eval()
        with torch.no_grad():
            band_tokens = self._band_power_tokens(eeg_signal)
            _, attention_weights = self.encoder(band_tokens, capture_attention=True)
            profile = attention_rollout_profile(attention_weights, has_cls_token=False)
        if was_training:
            self.train()
        return profile
