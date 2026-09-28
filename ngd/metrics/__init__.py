"""Evaluation distances between graphs and the mean pairwise diversity of a set."""

from __future__ import annotations

import networkx as nx
import numpy as np

from ngd.metrics import gcd, netlsd, portrait
from ngd.metrics.gcd import DEFAULT_ORCA_PATH

METRICS = ("gcd", "netlsd_heat", "netlsd_wave", "portrait_div")


def embed(
    graphs: list[nx.Graph], metric: str, orca_path: str = DEFAULT_ORCA_PATH, n_jobs: int = -1
) -> np.ndarray:
    if metric == "gcd":
        return np.array(gcd.gcd_embeddings(graphs, orca_path, n_jobs=n_jobs))
    if metric in ("netlsd_heat", "netlsd_wave"):
        return np.array(netlsd.signatures(graphs, metric.removeprefix("netlsd_")))
    raise ValueError(f"No vector embedding for metric {metric!r}.")


def pairwise_distances(
    graphs: list[nx.Graph], metric: str, orca_path: str = DEFAULT_ORCA_PATH, n_jobs: int = -1
) -> np.ndarray:
    """Symmetric matrix of ``metric`` distances between all pairs of ``graphs``."""
    n = len(graphs)
    dist = np.zeros((n, n), dtype=float)
    if metric == "portrait_div":
        vecs = portrait.portrait_distributions(portrait.portraits(graphs, n_jobs=n_jobs))
        pair = portrait.js_divergence
    else:
        vecs = list(embed(graphs, metric, orca_path, n_jobs=n_jobs))

        def pair(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.linalg.norm(x - y))

    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = pair(vecs[i], vecs[j])
    return dist


def diversity(
    graphs: list[nx.Graph], metric: str, orca_path: str = DEFAULT_ORCA_PATH, n_jobs: int = -1
) -> float:
    """Mean pairwise ``metric`` distance of a set of graphs."""
    dist = pairwise_distances(graphs, metric, orca_path, n_jobs=n_jobs)
    return float(np.mean(dist[np.triu_indices_from(dist, k=1)]))
