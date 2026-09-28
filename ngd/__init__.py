"""Neural Dispersion on Graphs (NGD): generation of maximally diverse graph sets."""

from __future__ import annotations

import networkx as nx

from ngd.config import NGDConfig, load_config
from ngd.metrics import diversity, pairwise_distances
from ngd.pipeline import run_pipeline
from ngd.selection import SelectionResult

__all__ = [
    "NGDConfig",
    "SelectionResult",
    "diversity",
    "generate",
    "load_config",
    "pairwise_distances",
    "run_pipeline",
]


def generate(n_vertices: int, k: int, metric: str, **kwargs) -> list[nx.Graph]:
    """Generate ``k`` graphs on ``n_vertices`` vertices that are diverse under ``metric``."""
    kwargs.setdefault("output_dir", None)
    config = NGDConfig(n_vertices=n_vertices, k_select=k, metrics=[metric], **kwargs)
    return run_pipeline(config)[metric].graphs
