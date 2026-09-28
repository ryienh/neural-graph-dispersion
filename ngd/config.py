"""Run configuration and YAML loading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from ngd.features import OPTIONAL_FEATURES
from ngd.metrics import DEFAULT_ORCA_PATH, METRICS


@dataclass
class NGDConfig:
    """All settings of one NGD run. Defaults correspond to ``configs/default.yaml``."""

    n_vertices: int = 16
    k_select: int = 100
    metrics: list[str] = field(default_factory=lambda: list(METRICS))

    num_ensembles: int = 80
    hidden_dim: int = 256
    num_hidden: int = 6
    latent_dim: int = 16

    batch_size: int = 100
    num_iterations: int = 5000
    learning_rate: float = 0.00023995817958282425
    num_samples_ste: int = 25
    gamma: float = 0.2
    epsilon: float = 0.001
    training_budget: int = 10_000

    features: list[str] = field(
        default_factory=lambda: ["regularity_proxy", "clustering_proxy", "adj_m3_norm", "adj_m2_norm"]
    )
    projection_dim: int = 4
    direction_seed: int = 42

    sampling_budget: int = 100_000
    n_top_matrices: int = 1000
    temp_range: tuple[float, float] = (0.01, 10.0)
    rigid_prob: float = 0.1
    memory_efficient: bool | None = None
    chunk_size: int | None = None

    seed: int | None = None
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    n_jobs: int = -1
    orca_path: str = DEFAULT_ORCA_PATH
    output_dir: str | None = "results"
    verbose: bool = True

    def __post_init__(self):
        self.temp_range = tuple(self.temp_range)
        bad = set(self.metrics) - set(METRICS)
        if bad:
            raise ValueError(f"Unknown metrics {sorted(bad)}; choose from {METRICS}.")
        bad = set(self.features) - set(OPTIONAL_FEATURES)
        if bad:
            raise ValueError(f"Unknown features {sorted(bad)}; choose from {OPTIONAL_FEATURES}.")
        if len(self.snapshot_iterations) * self.batch_size < self.graphs_per_ensemble:
            raise ValueError(
                "num_iterations is too small to collect graphs_per_ensemble snapshots per generator."
            )
        if self.memory_efficient is None:
            self.memory_efficient = self.n_vertices >= 256
        if self.memory_efficient and self.chunk_size is None:
            self.chunk_size = _default_chunk_size(self.n_vertices)

    @property
    def graphs_per_ensemble(self) -> int:
        return int(self.training_budget / self.num_ensembles)

    @property
    def snapshot_iterations(self) -> set[int]:
        snapshots = int(np.ceil(self.graphs_per_ensemble / self.batch_size))
        interval = max(1, self.num_iterations // snapshots)
        last = self.num_iterations - 1
        return {i for i in range(1, self.num_iterations) if i % interval == 0 or i == last}


def _default_chunk_size(n_vertices: int) -> int:
    base = max(10, (2 * 1024**3) // (n_vertices * n_vertices * 4))
    if n_vertices <= 256:
        return min(base, 500)
    if n_vertices <= 512:
        return min(base, 200)
    if n_vertices <= 1024:
        return min(base, 100)
    return min(base, 50)


def config_from_dict(raw: dict, **overrides) -> NGDConfig:
    raw = {**raw, **overrides}
    fields = {f.name: f for f in dataclasses.fields(NGDConfig)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ValueError(f"Unknown config keys {sorted(unknown)}; valid keys: {sorted(fields)}.")
    for key, value in raw.items():
        if fields[key].type in ("float", "int") and isinstance(value, str):
            raw[key] = float(value) if fields[key].type == "float" else int(float(value))
    return NGDConfig(**raw)


def load_config(path: str | Path, **overrides) -> NGDConfig:
    """Load a YAML config; keyword arguments override values from the file."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return config_from_dict(raw, **overrides)
