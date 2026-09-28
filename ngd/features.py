"""Differentiable spectral-moment features and the Coulomb dispersion loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPSILON = 1e-8

CORE_FEATURES = (
    "adj_m3_m2",
    "adj_m4_m2",
    "adj_m5_m3",
    "adj_m6_m4",
    "adj_m4_m3",
    "adj_m6_m2",
)
OPTIONAL_FEATURES = (
    "adj_m5_m2",
    "adj_m6_m3",
    "adj_m5_m4",
    "regularity_proxy",
    "spectral_spread",
    "clustering_proxy",
    "triangle_density",
    "adj_m3_norm",
    "adj_m4_norm",
    "adj_m2_norm",
)
FEATURE_ORDER = CORE_FEATURES + OPTIONAL_FEATURES


def _moments(adj: torch.Tensor, max_power: int = 6) -> dict[int, torch.Tensor]:
    moments = {}
    a2 = torch.bmm(adj, adj)
    moments[2] = torch.diagonal(a2, dim1=1, dim2=2).sum(dim=1)
    ak = a2
    for k in range(3, max_power + 1):
        ak = torch.bmm(ak, adj)
        moments[k] = torch.diagonal(ak, dim1=1, dim2=2).sum(dim=1)
    return moments


def _safe_ratio(num: torch.Tensor, denom: torch.Tensor, max_val: float = 100.0) -> torch.Tensor:
    ratio = num / (denom.abs() + EPSILON)
    mask = (denom.abs() > EPSILON * 100).float()
    return (ratio * mask).clamp(-max_val, max_val)


def _normalize(x: torch.Tensor, expected_max: float) -> torch.Tensor:
    return (x / (expected_max + EPSILON)).clamp(0, 1)


def spectral_features(adj: torch.Tensor, names: list[str]) -> torch.Tensor:
    n = adj.shape[1]
    m = _moments(adj)
    f = {
        "adj_m3_m2": _normalize(_safe_ratio(m[3], m[2]), n),
        "adj_m4_m2": _normalize(_safe_ratio(m[4], m[2]), n * n),
        "adj_m5_m3": _normalize(_safe_ratio(m[5], m[3]), n * n),
        "adj_m6_m4": _normalize(_safe_ratio(m[6], m[4]), n * n),
        "adj_m4_m3": _normalize(_safe_ratio(m[4], m[3]), n),
        "adj_m6_m2": _normalize(_safe_ratio(m[6], m[2]), n**4),
        "adj_m5_m2": _normalize(_safe_ratio(m[5], m[2]), n**3),
        "adj_m6_m3": _normalize(_safe_ratio(m[6], m[3]), n**3),
        "adj_m5_m4": _normalize(_safe_ratio(m[5], m[4]), n),
        "regularity_proxy": _safe_ratio(m[2] ** 2, n * m[4]).clamp(0, 2) / 2,
        "spectral_spread": _normalize(_safe_ratio(m[4], m[2] ** 2 + EPSILON), n),
        "clustering_proxy": _safe_ratio(m[3] ** 2, m[6]).clamp(0, 2) / 2,
        "triangle_density": _normalize(_safe_ratio(m[3], (m[2] + EPSILON) ** 1.5), n**0.5),
        "adj_m3_norm": (m[3] / (n**3 + EPSILON)).clamp(0, 1),
        "adj_m4_norm": (m[4] / (n**4 + EPSILON)).clamp(0, 1),
        "adj_m2_norm": (m[2] / (n * (n - 1) + EPSILON)).clamp(0, 1),
    }
    return torch.stack([f[name] for name in names], dim=1)


class SpectralProjection:
    def __init__(
        self,
        optional_features: list[str],
        num_ensembles: int,
        proj_dim: int,
        direction_seed: int,
    ):
        enabled = set(CORE_FEATURES) | set(optional_features)
        self.names = [name for name in FEATURE_ORDER if name in enabled]
        gen = torch.Generator().manual_seed(direction_seed)
        directions = torch.randn(num_ensembles, proj_dim, len(self.names), generator=gen)
        self.directions = directions / (directions.norm(dim=2, keepdim=True) + EPSILON)

    def to(self, device: str | torch.device) -> SpectralProjection:
        self.directions = self.directions.to(device)
        return self

    def __call__(self, batch: torch.Tensor, ensemble_id: int) -> torch.Tensor:
        n = batch.shape[1]
        adj = (batch + batch.transpose(1, 2)) / 2
        adj = adj * (1.0 - torch.eye(n, device=batch.device).unsqueeze(0))
        feats = spectral_features(adj, self.names)
        return torch.mm(feats, self.directions[ensemble_id % len(self.directions)].T)


def pairwise_distance(x: torch.Tensor) -> torch.Tensor:
    sq_norms = (x**2).sum(dim=1, keepdim=True)
    d2 = F.relu(sq_norms + sq_norms.t() - 2 * torch.mm(x, x.t()))
    return torch.sqrt(d2 + EPSILON)


def coulomb_loss(features: torch.Tensor, gamma: float, epsilon: float) -> torch.Tensor:
    b = features.shape[0]
    mask = 1.0 - torch.eye(b, device=features.device)
    potential = mask / (pairwise_distance(features) + epsilon) ** gamma
    return potential.sum() / (b * (b - 1))
