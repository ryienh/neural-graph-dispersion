"""Ensemble of MLP graph generators, evaluated in parallel with batched matmuls."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_ensembles: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensembles, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(num_ensembles, 1, out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weight) + self.bias


class EnsembleGraphGenerator(nn.Module):
    def __init__(
        self,
        num_ensembles: int,
        latent_dim: int,
        hidden_dim: int,
        num_hidden: int,
        n_vertices: int,
        num_particles: int,
    ):
        super().__init__()
        self.num_ensembles = num_ensembles
        self.n_vertices = n_vertices
        self.id_embedding = nn.Parameter(torch.randn(num_ensembles, num_particles, hidden_dim))
        self.z_project = EnsembleLinear(latent_dim, hidden_dim, num_ensembles)
        self.layers = nn.ModuleList(
            EnsembleLinear(hidden_dim, hidden_dim, num_ensembles) for _ in range(num_hidden)
        )
        self.output_layer = EnsembleLinear(hidden_dim, n_vertices * n_vertices, num_ensembles)

    def forward(self, z: torch.Tensor, particle_ids: torch.Tensor) -> torch.Tensor:
        e, b, n = self.num_ensembles, z.shape[0], self.n_vertices
        x = F.relu(self.z_project(z.unsqueeze(0).expand(e, b, -1)) + self.id_embedding[:, particle_ids, :])
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x).view(e, b, n, n)
        probs = torch.sigmoid((x + x.transpose(-1, -2)) / 2)
        return probs * (1 - torch.eye(n, device=x.device).unsqueeze(0).unsqueeze(0))
