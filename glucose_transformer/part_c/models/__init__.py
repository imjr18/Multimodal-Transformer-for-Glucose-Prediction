"""Efficient EEG encoders and the full Part C multimodal backbone."""

from .frequency_eeg import FrequencyEEGEncoder
from .full_modal import FullModalTransformer
from .hierarchical_eeg import HierarchicalEEGEncoder
from .patch_tst_eeg import PatchEEGEncoder

__all__ = [
    "FrequencyEEGEncoder",
    "FullModalTransformer",
    "HierarchicalEEGEncoder",
    "PatchEEGEncoder",
]
