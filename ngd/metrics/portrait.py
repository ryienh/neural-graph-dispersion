"""Portrait divergence (Bagrow & Bollt, 2019)."""

from __future__ import annotations

import networkx as nx
import numpy as np
from joblib import Parallel, delayed


def portrait(graph: nx.Graph) -> np.ndarray:
    n = graph.number_of_nodes()
    if n == 0:
        return np.array([[1.0]])
    try:
        diameter = nx.diameter(graph)
    except nx.NetworkXError:
        diameter = n
    b = np.zeros((diameter + 1, n))
    max_path = 1
    adj = graph.adj
    for source in graph.nodes():
        dist = {source: 0}
        frontier = [source]
        d = 1
        while frontier:
            nxt = []
            for u in frontier:
                new = [v for v in adj[u] if v not in dist]
                nxt.extend(new)
                for v in new:
                    dist[v] = d
            frontier = nxt
            d += 1
        shells = dist.values()
        farthest = max(shells)
        max_path = max(max_path, farthest)
        counts = dict.fromkeys(shells, 0)
        for s in shells:
            counts[s] += 1
        for shell, count in counts.items():
            if shell <= diameter and count < n:
                b[shell][count] += 1
        for shell in range(diameter, farthest, -1):
            b[shell][0] += 1
    return b[: max_path + 1, :]


def portraits(graphs: list[nx.Graph], n_jobs: int = -1) -> list[np.ndarray]:
    return Parallel(n_jobs=n_jobs, prefer="threads")(delayed(portrait)(g) for g in graphs)


def _distribution(b: np.ndarray) -> np.ndarray:
    n = int(b[0, 1]) if b.shape[1] > 1 and b[0, 1] > 0 else b.shape[1]
    f = (b * np.arange(b.shape[1])).sum(axis=1)
    p_l = f / f.sum() if f.sum() > 0 else np.zeros(b.shape[0])
    p_k_given_l = b / n if n > 0 else b
    return (p_k_given_l * p_l[:, None]).ravel()


def portrait_distributions(portraits_: list[np.ndarray]) -> list[np.ndarray]:
    rows = max(b.shape[0] for b in portraits_)
    cols = 0
    for b in portraits_:
        nz = np.nonzero(b)[1]
        if len(nz) > 0:
            cols = max(cols, int(np.max(nz)) + 1)
    cols = max(cols, 1)
    out = []
    for b in portraits_:
        padded = np.zeros((rows, cols))
        c = min(b.shape[1], cols)
        padded[: b.shape[0], :c] = b[:, :c]
        out.append(_distribution(padded))
    return out


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    mp, mq = p > 0, q > 0
    kl_pm = np.sum(p[mp] * np.log2(p[mp] / m[mp]))
    kl_qm = np.sum(q[mq] * np.log2(q[mq] / m[mq]))
    return 0.5 * (kl_pm + kl_qm)
