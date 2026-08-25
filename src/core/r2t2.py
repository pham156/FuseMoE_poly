from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _sample_correct_mask(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    logits_arr = np.asarray(logits)
    labels_arr = np.asarray(labels)
    if logits_arr.ndim == 1:
        pred = (logits_arr > 0.5).astype(labels_arr.dtype)
        return pred.reshape(-1) == labels_arr.reshape(-1)
    if logits_arr.shape[-1] == 2 and labels_arr.ndim == 1:
        pred = np.argmax(logits_arr, axis=-1)
        return pred.reshape(-1) == labels_arr.reshape(-1)
    pred = (logits_arr > 0.5).astype(labels_arr.dtype)
    return np.all(pred == labels_arr, axis=tuple(range(1, pred.ndim)))


class R2T2ReferenceCollector:
    """Collect validation router states for test-time rerouting."""

    def __init__(self, correct_reference_only: bool = False):
        self.correct_reference_only = bool(correct_reference_only)
        self._features: dict[tuple[int, int], list[torch.Tensor]] = {}
        self._gates: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.num_batches = 0
        self.num_samples_seen = 0
        self.num_samples_kept = 0

    def add_batch(
        self,
        diagnostics_list: list[dict[str, Any]],
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if not diagnostics_list:
            return
        keep_mask = np.ones(len(labels), dtype=bool)
        if self.correct_reference_only:
            keep_mask = _sample_correct_mask(logits, labels)
        keep_idx = torch.as_tensor(np.nonzero(keep_mask)[0], dtype=torch.long)
        self.num_batches += 1
        self.num_samples_seen += int(len(labels))
        self.num_samples_kept += int(keep_idx.numel())
        if keep_idx.numel() == 0:
            return

        for layer_idx, diagnostics in enumerate(diagnostics_list):
            router_inputs = _as_list(diagnostics.get("router_inputs"))
            gates = _as_list(diagnostics.get("gates"))
            for modality_idx, (features, gate) in enumerate(zip(router_inputs, gates)):
                if not isinstance(features, torch.Tensor) or not isinstance(gate, torch.Tensor):
                    continue
                if features.ndim != 2 or gate.ndim != 2:
                    continue
                valid_idx = keep_idx[keep_idx < min(features.size(0), gate.size(0))]
                if valid_idx.numel() == 0:
                    continue
                key = (layer_idx, modality_idx)
                self._features.setdefault(key, []).append(features.detach().float().cpu()[valid_idx])
                self._gates.setdefault(key, []).append(gate.detach().float().cpu()[valid_idx])

    def build(self, num_neighbors: int, blend: float, kernel_sigma: float) -> "R2T2Adaptor":
        bank: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for key, feature_parts in self._features.items():
            gate_parts = self._gates.get(key, [])
            if not feature_parts or not gate_parts:
                continue
            features = torch.cat(feature_parts, dim=0)
            gates = torch.cat(gate_parts, dim=0)
            if features.numel() == 0 or gates.numel() == 0:
                continue
            bank[key] = (F.normalize(features, dim=-1), gates)
        return R2T2Adaptor(
            bank=bank,
            num_neighbors=num_neighbors,
            blend=blend,
            kernel_sigma=kernel_sigma,
            num_samples_seen=self.num_samples_seen,
            num_samples_kept=self.num_samples_kept,
        )


@dataclass
class R2T2Adaptor:
    bank: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]
    num_neighbors: int = 32
    blend: float = 0.5
    kernel_sigma: float = 1.0
    num_samples_seen: int = 0
    num_samples_kept: int = 0

    def __post_init__(self) -> None:
        self.num_neighbors = max(1, int(self.num_neighbors))
        self.blend = float(min(max(self.blend, 0.0), 1.0))
        self.kernel_sigma = max(float(self.kernel_sigma), 1e-6)
        self.applied_batches = 0
        self.unchanged_batches = 0

    @property
    def num_reference_groups(self) -> int:
        return len(self.bank)

    @property
    def num_reference_vectors(self) -> int:
        return int(sum(features.size(0) for features, _ in self.bank.values()))

    def _reroute_one(
        self,
        layer_idx: int,
        modality_idx: int,
        features: torch.Tensor,
        gates: torch.Tensor,
    ) -> torch.Tensor:
        key = (layer_idx, modality_idx)
        if key not in self.bank or features.ndim != 2 or gates.ndim != 2:
            self.unchanged_batches += 1
            return gates

        ref_features, ref_gates = self.bank[key]
        if ref_features.size(0) == 0:
            self.unchanged_batches += 1
            return gates

        device = gates.device
        current = F.normalize(features.detach().float().to(device), dim=-1)
        ref_features = ref_features.to(device)
        ref_gates = ref_gates.to(device=device, dtype=gates.dtype)
        k = min(self.num_neighbors, ref_features.size(0))
        distances = torch.cdist(current, ref_features)
        knn_dist, knn_idx = torch.topk(distances, k=k, dim=1, largest=False)
        weights = torch.softmax(-(knn_dist ** 2) / (2.0 * self.kernel_sigma ** 2), dim=1)
        neighbor_gates = torch.sum(ref_gates[knn_idx] * weights.unsqueeze(-1), dim=1)
        blended = (1.0 - self.blend) * gates + self.blend * neighbor_gates

        active_k = int((gates > 0).sum(dim=1).max().detach().cpu().item())
        active_k = max(1, min(active_k, gates.size(1)))
        top_values, top_indices = torch.topk(blended, k=active_k, dim=1)
        sparse = torch.zeros_like(gates)
        sparse.scatter_(1, top_indices, top_values.clamp_min(0.0))
        sparse = sparse / sparse.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self.applied_batches += 1
        return sparse

    def reroute(self, layer_idx: int, router_inputs: Any, gates: Any) -> Any:
        if isinstance(gates, list):
            inputs = _as_list(router_inputs)
            output = []
            for modality_idx, gate in enumerate(gates):
                features = inputs[modality_idx] if modality_idx < len(inputs) else None
                if isinstance(features, torch.Tensor):
                    output.append(self._reroute_one(layer_idx, modality_idx, features, gate))
                else:
                    self.unchanged_batches += 1
                    output.append(gate)
            return output
        modality_idx = 0
        features = router_inputs[0] if isinstance(router_inputs, list) and router_inputs else router_inputs
        if isinstance(features, torch.Tensor):
            return self._reroute_one(layer_idx, modality_idx, features, gates)
        self.unchanged_batches += 1
        return gates
