"""User-conditioning modules that wrap the Part C backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from part_b.models.common import SequenceEncoder
from part_c.models.common import TrackedSequenceEncoder
from part_c.models.full_modal import FullModalTransformer


@dataclass
class UserConditioningContext:
    """Mutable context shared by wrapped encoder layers during one forward pass."""

    current_embedding: torch.Tensor | None = None


class UserConditionedEncoderLayer(nn.Module):
    """Wrap one encoder layer and condition the CLS token on user identity.

    The wrapped layer remains the original Part C layer. This wrapper only
    modifies the first token immediately before attention by concatenating the
    current user embedding and projecting back to `d_model`, which satisfies the
    Part D requirement without changing the backbone topology.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        context: UserConditioningContext,
        embedding_dim: int,
        d_model: int,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.context = context
        self.condition_projection = nn.Linear(d_model + embedding_dim, d_model)

    def __getattr__(self, name: str):
        """Proxy unknown attributes to the wrapped encoder layer."""

        try:
            return super().__getattr__(name)
        except AttributeError as error:
            base_layer = self._modules.get("base_layer")
            if base_layer is not None and hasattr(base_layer, name):
                return getattr(base_layer, name)
            raise error

    @property
    def capture_attention(self):
        """Proxy attention-capture flags to the wrapped layer."""

        return getattr(self.base_layer, "capture_attention", False)

    @capture_attention.setter
    def capture_attention(self, value) -> None:
        setattr(self.base_layer, "capture_attention", value)

    @property
    def latest_attention_weights(self):
        """Expose the wrapped layer's last attention map."""

        return getattr(self.base_layer, "latest_attention_weights", None)

    def _condition_cls_token(self, src: torch.Tensor) -> torch.Tensor:
        """Inject the current user embedding into the first token."""

        user_embedding = self.context.current_embedding
        if user_embedding is None or src.size(1) == 0:
            return src

        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)
        if user_embedding.size(0) == 1 and src.size(0) > 1:
            user_embedding = user_embedding.expand(src.size(0), -1)

        conditioned_cls = self.condition_projection(
            torch.cat(
                [src[:, :1, :], user_embedding.to(device=src.device, dtype=src.dtype).unsqueeze(1)],
                dim=-1,
            )
        )
        if src.size(1) == 1:
            return conditioned_cls
        return torch.cat([conditioned_cls, src[:, 1:, :]], dim=1)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Run the original encoder layer on a user-conditioned sequence."""

        conditioned_src = self._condition_cls_token(src)
        return self.base_layer(
            conditioned_src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )


class UserEmbeddingModule(nn.Module):
    """Learnable identity lookup table for known training users."""

    def __init__(self, n_users: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_users, embedding_dim)
        self.register_buffer("known_user_mask", torch.zeros(n_users, dtype=torch.bool), persistent=False)

    def set_known_user_ids(self, user_ids: list[int]) -> None:
        """Mark which user IDs may use direct lookup embeddings."""

        mask = torch.zeros_like(self.known_user_mask)
        if user_ids:
            indices = torch.as_tensor(user_ids, dtype=torch.long)
            mask[indices] = True
        self.known_user_mask.copy_(mask)

    def lookup(self, user_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lookup embeddings and a mask of which IDs are known."""

        user_ids = user_ids.to(device=self.embedding.weight.device, dtype=torch.long)
        safe_user_ids = user_ids.clamp(min=0, max=self.embedding.num_embeddings - 1)
        embeddings = self.embedding(safe_user_ids)
        known_mask = self.known_user_mask[safe_user_ids]
        return embeddings, known_mask

    def all_embeddings(self) -> torch.Tensor:
        """Return the full user embedding table."""

        return self.embedding.weight.detach().cpu()


class ArchetypeEmbeddingModule(nn.Module):
    """Prototype embeddings used to warm-start previously unseen users."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(4, embedding_dim)

    def forward(self, archetype_ids: torch.Tensor) -> torch.Tensor:
        """Return one archetype prototype per batch element."""

        return self.embedding(archetype_ids.to(dtype=torch.long))

    def all_embeddings(self) -> torch.Tensor:
        """Return the archetype prototype table."""

        return self.embedding.weight.detach().cpu()


class UserConditionedFullModalTransformer(nn.Module):
    """Wrap the Part C multimodal backbone with user conditioning.

    The wrapper owns the original Part C `FullModalTransformer` unchanged and
    injects user embeddings into the CLS token before every relevant encoder
    layer. Known training users use a learned lookup vector; unseen users fall
    back to an archetype prototype unless an explicit per-user override is
    provided during adaptation.
    """

    def __init__(
        self,
        config: dict,
        *,
        eeg_encoder_kind: str,
        n_users: int,
        inject_conditioning: bool = True,
    ):
        super().__init__()
        self.config = config
        self.embedding_dim = int(config["user_embedding_dim"])
        self.conditioning_context = UserConditioningContext()

        self.backbone = FullModalTransformer(config, eeg_encoder_kind=eeg_encoder_kind)
        self.user_embedding_module = UserEmbeddingModule(n_users=n_users, embedding_dim=self.embedding_dim)
        self.archetype_embedding_module = ArchetypeEmbeddingModule(self.embedding_dim)
        if inject_conditioning:
            self.inject_user_conditioning()

    def _wrap_transformer_encoder(self, encoder: nn.TransformerEncoder, *, has_cls_token: bool) -> None:
        """Replace encoder layers with user-conditioned wrappers when appropriate."""

        if not has_cls_token:
            return
        if not encoder.layers:
            return
        if isinstance(encoder.layers[0], UserConditionedEncoderLayer):
            return

        d_model = int(self.config["d_model"])
        encoder.layers = nn.ModuleList(
            [
                UserConditionedEncoderLayer(
                    layer,
                    self.conditioning_context,
                    embedding_dim=self.embedding_dim,
                    d_model=d_model,
                )
                for layer in encoder.layers
            ]
        )

    def inject_user_conditioning(self) -> None:
        """Attach user-conditioned wrappers to the backbone's encoder stacks."""

        for module in self.backbone.modules():
            if isinstance(module, SequenceEncoder):
                self._wrap_transformer_encoder(module.encoder, has_cls_token=True)
            elif isinstance(module, TrackedSequenceEncoder):
                self._wrap_transformer_encoder(module.encoder, has_cls_token=bool(module.use_cls_token))

        self._wrap_transformer_encoder(self.backbone.final_fusion_encoder, has_cls_token=True)

    def set_known_user_ids(self, user_ids: list[int]) -> None:
        """Configure which user IDs should use the lookup table."""

        self.user_embedding_module.set_known_user_ids(user_ids)

    def _resolve_archetype_ids(self, batch_size: int, archetype_ids) -> torch.Tensor:
        """Normalise optional archetype IDs to a batch-long tensor."""

        if archetype_ids is None:
            return torch.zeros(batch_size, dtype=torch.long)
        if isinstance(archetype_ids, torch.Tensor):
            tensor = archetype_ids.to(dtype=torch.long)
        else:
            tensor = torch.as_tensor(archetype_ids, dtype=torch.long)
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        if tensor.size(0) == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size)
        return tensor

    def resolve_user_embedding(
        self,
        *,
        batch_size: int,
        user_ids=None,
        archetype_ids=None,
        user_embedding_override: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Resolve the per-sample user embedding used in the current forward pass."""

        if user_embedding_override is not None:
            override = user_embedding_override
            if override.dim() == 1:
                override = override.unsqueeze(0)
            if override.size(0) == 1 and batch_size > 1:
                override = override.expand(batch_size, -1)
            return override.to(device=device)

        resolved_archetype_ids = self._resolve_archetype_ids(batch_size, archetype_ids)
        resolved_archetype_ids = resolved_archetype_ids.to(device=device)
        archetype_embeddings = self.archetype_embedding_module(resolved_archetype_ids)

        if user_ids is None:
            return archetype_embeddings

        user_id_tensor = user_ids if isinstance(user_ids, torch.Tensor) else torch.as_tensor(user_ids, dtype=torch.long)
        if user_id_tensor.dim() == 0:
            user_id_tensor = user_id_tensor.unsqueeze(0)
        if user_id_tensor.size(0) == 1 and batch_size > 1:
            user_id_tensor = user_id_tensor.expand(batch_size)
        user_id_tensor = user_id_tensor.to(device=device, dtype=torch.long)

        lookup_embeddings, known_mask = self.user_embedding_module.lookup(user_id_tensor)
        lookup_embeddings = lookup_embeddings.to(device=device)
        known_mask = known_mask.to(device=device)
        return torch.where(known_mask.unsqueeze(-1), lookup_embeddings, archetype_embeddings)

    def forward(
        self,
        hr_sequence: torch.Tensor,
        glucose_context: torch.Tensor,
        ecg_features: torch.Tensor,
        emg_features: torch.Tensor,
        eeg_signal: torch.Tensor,
        cbf_signal: torch.Tensor,
        *,
        user_ids=None,
        archetype_ids=None,
        user_embedding_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the Part C backbone with an active user-conditioning context."""

        user_embedding = self.resolve_user_embedding(
            batch_size=hr_sequence.size(0),
            user_ids=user_ids,
            archetype_ids=archetype_ids,
            user_embedding_override=user_embedding_override,
            device=hr_sequence.device,
        )
        self.conditioning_context.current_embedding = user_embedding
        try:
            return self.backbone(
                hr_sequence,
                glucose_context,
                ecg_features,
                emg_features,
                eeg_signal,
                cbf_signal,
            )
        finally:
            self.conditioning_context.current_embedding = None

    def get_initial_user_embedding(self, user_id: int, archetype_id: int, *, device: str | torch.device) -> torch.Tensor:
        """Return the starting embedding for adaptation or analysis."""

        embedding = self.resolve_user_embedding(
            batch_size=1,
            user_ids=torch.tensor([user_id], dtype=torch.long, device=device),
            archetype_ids=torch.tensor([archetype_id], dtype=torch.long, device=device),
            device=device,
        )
        return embedding.squeeze(0)


__all__ = [
    "ArchetypeEmbeddingModule",
    "UserConditionedEncoderLayer",
    "UserConditionedFullModalTransformer",
    "UserEmbeddingModule",
]
