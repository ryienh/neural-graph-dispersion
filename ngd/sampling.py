"""Sampling discrete graphs from edge-probability matrices."""

from __future__ import annotations

import networkx as nx
import numpy as np

_CLIP = 1e-6


def sample_bernoulli(
    probs: np.ndarray, temp: float = 1.0, random_state: np.random.RandomState | None = None
) -> np.ndarray:
    probs = np.array(probs, copy=True)
    probs = (probs + probs.T) / 2.0
    np.fill_diagonal(probs, 0.0)
    if abs(temp - 1.0) > 1e-4:
        p = np.clip(probs, _CLIP, 1.0 - _CLIP)
        logits = np.log(p / (1.0 - p)) / temp
        with np.errstate(over="ignore"):
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
    rand = (np.random if random_state is None else random_state).rand
    adj = (rand(*probs.shape) < probs).astype(float)
    upper = np.triu(adj, k=1)
    return upper + upper.T


def sample_rigid(probs: np.ndarray) -> np.ndarray:
    """Deterministic graph keeping the k most likely edges, k = expected edge count."""
    probs = np.array(probs, copy=True)
    probs = (probs + probs.T) / 2.0
    np.fill_diagonal(probs, 0.0)
    mask = np.triu(np.ones_like(probs), k=1).astype(bool)
    p_upper = probs[mask]
    k = int(np.round(np.sum(p_upper)))
    if k == 0:
        return np.zeros_like(probs)
    adj_upper = np.zeros_like(p_upper)
    adj_upper[np.argsort(p_upper)[-k:]] = 1.0
    adj = np.zeros_like(probs)
    adj[mask] = adj_upper
    return adj + adj.T


def sample(probs: np.ndarray, temp: float, random_state: np.random.RandomState | None = None) -> np.ndarray:
    if np.isinf(temp):
        return sample_rigid(probs)
    return sample_bernoulli(probs, temp, random_state)


def adj_to_nx(adj: np.ndarray) -> nx.Graph:
    rows, cols = np.where(adj > 0.5)
    graph = nx.Graph()
    graph.add_nodes_from(range(adj.shape[0]))
    graph.add_edges_from(zip(rows.tolist(), cols.tolist(), strict=True))
    return graph
