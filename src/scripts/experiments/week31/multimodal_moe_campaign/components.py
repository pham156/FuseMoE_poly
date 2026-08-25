from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_ORDER = ("text", "ts", "cxr", "ecg")
PAIR_ORDER = (
    ("text", "ts"),
    ("text", "cxr"),
    ("text", "ecg"),
    ("ts", "cxr"),
    ("ts", "ecg"),
    ("cxr", "ecg"),
)


def _masked_zero(vec: torch.Tensor, missing: Optional[torch.Tensor]) -> torch.Tensor:
    if missing is None:
        return vec
    mask = missing.to(device=vec.device).bool().view(-1, 1)
    return vec.masked_fill(mask.expand_as(vec), 0.0)


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


@dataclass
class InteractionFeatureOutput:
    features: torch.Tensor
    feature_names: List[str]
    feature_dict: Dict[str, torch.Tensor]


class InteractionFeatureExtractor(nn.Module):
    def __init__(
        self,
        include_l2: bool = False,
        include_dot: bool = True,
        include_norm_ratio: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.include_l2 = include_l2
        self.include_dot = include_dot
        self.include_norm_ratio = include_norm_ratio
        self.eps = eps

    def forward(
        self,
        pooled_embeddings: Dict[str, Optional[torch.Tensor]],
        missing_flags: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    ) -> InteractionFeatureOutput:
        missing_flags = missing_flags or {}
        batch_size = None
        device = None
        dtype = None
        for key in MODALITY_ORDER:
            value = pooled_embeddings.get(key)
            if value is not None:
                batch_size = value.size(0)
                device = value.device
                dtype = value.dtype
                break
        if batch_size is None:
            raise ValueError("InteractionFeatureExtractor requires at least one modality embedding.")

        features: List[torch.Tensor] = []
        feature_names: List[str] = []
        feature_dict: Dict[str, torch.Tensor] = {}

        def zero_feature() -> torch.Tensor:
            return torch.zeros(batch_size, 1, device=device, dtype=dtype)

        for left_name, right_name in PAIR_ORDER:
            left = pooled_embeddings.get(left_name)
            right = pooled_embeddings.get(right_name)
            if left is None or right is None:
                pair_missing = torch.ones(batch_size, dtype=torch.bool, device=device)
                cosine = zero_feature()
                l2 = zero_feature()
                dot = zero_feature()
                ratio = zero_feature()
            else:
                left_missing = missing_flags.get(left_name)
                right_missing = missing_flags.get(right_name)
                pair_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)
                if left_missing is not None:
                    pair_missing |= left_missing.to(device=device).bool().view(-1)
                if right_missing is not None:
                    pair_missing |= right_missing.to(device=device).bool().view(-1)
                left_masked = _masked_zero(left, pair_missing)
                right_masked = _masked_zero(right, pair_missing)
                cosine = F.cosine_similarity(left_masked, right_masked, dim=-1, eps=self.eps).unsqueeze(1)
                l2 = torch.norm(left_masked - right_masked, dim=-1, p=2).unsqueeze(1)
                dot_raw = (left_masked * right_masked).sum(dim=-1)
                dot = _signed_log1p(dot_raw).unsqueeze(1)
                ratio = (
                    torch.norm(left_masked, dim=-1, p=2)
                    / torch.clamp(torch.norm(right_masked, dim=-1, p=2), min=self.eps)
                ).unsqueeze(1)
                cosine = torch.nan_to_num(cosine, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
                l2 = torch.nan_to_num(l2, nan=0.0, posinf=0.0, neginf=0.0)
                dot = torch.nan_to_num(dot, nan=0.0, posinf=0.0, neginf=0.0)
                ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
                cosine[pair_missing] = 0.0
                l2[pair_missing] = 0.0
                dot[pair_missing] = 0.0
                ratio[pair_missing] = 0.0

            name = f"cosine_{left_name}_{right_name}"
            features.append(cosine)
            feature_names.append(name)
            feature_dict[name] = cosine.detach()

            if self.include_l2:
                name = f"l2_{left_name}_{right_name}"
                features.append(l2)
                feature_names.append(name)
                feature_dict[name] = l2.detach()

            if self.include_dot:
                name = f"dot_{left_name}_{right_name}"
                features.append(dot)
                feature_names.append(name)
                feature_dict[name] = dot.detach()

            if self.include_norm_ratio:
                name = f"norm_ratio_{left_name}_{right_name}"
                features.append(ratio)
                feature_names.append(name)
                feature_dict[name] = ratio.detach()

        return InteractionFeatureOutput(
            features=torch.cat(features, dim=1),
            feature_names=feature_names,
            feature_dict=feature_dict,
        )


class MissingnessEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        encoder_type: str = "lookup",
        num_modalities: int = 4,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.encoder_type = encoder_type
        self.num_modalities = int(num_modalities)
        self.num_patterns = 2 ** self.num_modalities

        if encoder_type == "lookup":
            self.encoder = nn.Embedding(self.num_patterns, self.output_dim)
            nn.init.normal_(self.encoder.weight, std=0.02)
        elif encoder_type == "linear":
            self.encoder = nn.Linear(self.num_modalities, self.output_dim)
            nn.init.xavier_uniform_(self.encoder.weight)
            nn.init.zeros_(self.encoder.bias)
        elif encoder_type == "mlp":
            hidden = int(hidden_dim or max(self.output_dim, 32))
            self.encoder = nn.Sequential(
                nn.Linear(self.num_modalities, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.output_dim),
            )
            for layer in self.encoder:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unknown missingness encoder type: {encoder_type}")

    def forward(self, observed_flags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        observed_flags = observed_flags.float()
        if observed_flags.dim() != 2 or observed_flags.size(1) != self.num_modalities:
            raise ValueError(
                f"Expected observed_flags shape [batch,{self.num_modalities}], got {tuple(observed_flags.shape)}"
            )
        mask_ids = (
            observed_flags[:, 0].long()
            + 2 * observed_flags[:, 1].long()
            + 4 * observed_flags[:, 2].long()
            + 8 * observed_flags[:, 3].long()
        )
        if self.encoder_type == "lookup":
            return self.encoder(mask_ids), mask_ids
        return self.encoder(observed_flags), mask_ids


class SharedSemanticMemory(nn.Module):
    def __init__(self, dim: int, num_slots: int = 8, mode: str = "learnable", num_heads: int = 1):
        super().__init__()
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.mode = mode
        self.num_heads = int(max(1, num_heads))

        memory = torch.randn(self.num_slots, self.dim) * 0.02
        if mode == "none":
            self.register_buffer("memory", memory)
            self.memory.requires_grad = False
        elif mode == "fixed":
            self.register_buffer("memory", memory)
        elif mode == "learnable":
            self.memory = nn.Parameter(memory)
        else:
            raise ValueError(f"Unknown semantic memory mode: {mode}")

        self.input_norm = nn.LayerNorm(self.dim)
        self.memory_norm = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads, batch_first=True)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        orig_dtype = x.dtype
        memory = self.memory
        if not isinstance(memory, torch.Tensor):
            memory = memory.data
        memory = torch.nan_to_num(
            memory.to(device=x.device, dtype=x.dtype),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        compute_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        query = self.input_norm(x.to(dtype=compute_dtype)).unsqueeze(1)
        key = self.memory_norm(memory.to(dtype=compute_dtype)).unsqueeze(0).expand(x.size(0), -1, -1)
        value = key
        with torch.autocast(device_type=x.device.type, enabled=False):
            attended, _ = self.attn(query, key, value, need_weights=False)
            attended = torch.nan_to_num(attended, nan=0.0, posinf=0.0, neginf=0.0)
            projected = self.out_proj(attended.squeeze(1))
            projected = torch.nan_to_num(projected, nan=0.0, posinf=0.0, neginf=0.0)
        projected = torch.tanh(projected) * torch.clamp(self.residual_scale, min=0.0, max=1.0)
        projected = projected.to(dtype=orig_dtype)
        return torch.nan_to_num(x + projected, nan=0.0, posinf=0.0, neginf=0.0)


class TaskSpecificRouter(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, task_names: List[str]):
        super().__init__()
        self.task_names = [str(name).lower() for name in task_names]
        self.task_to_id = {name: idx for idx, name in enumerate(self.task_names)}
        self.routers = nn.ModuleDict({
            name: nn.Linear(input_dim, num_experts, bias=False)
            for name in self.task_names
        })
        for layer in self.routers.values():
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        key = str(task_name).lower()
        if key not in self.routers:
            raise KeyError(f"Unknown task router '{task_name}'. Available: {sorted(self.routers.keys())}")
        return self.routers[key](x)
