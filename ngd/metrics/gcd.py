"""Graphlet correlation distance (Yaveroglu et al., 2014), via ORCA orbit counts."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata

DEFAULT_ORCA_PATH = str(Path(__file__).resolve().parent / "orca" / "orca")

_ORBITS = [0, 2, 5, 7, 8, 10, 11, 6, 9, 4, 1]


def _orbit_counts(graph: nx.Graph, orca_path: str = DEFAULT_ORCA_PATH) -> np.ndarray:
    if not os.path.exists(orca_path):
        raise FileNotFoundError(f"ORCA binary not found at {orca_path}. See the README for how to build it.")
    index = {u: i for i, u in enumerate(graph.nodes())}
    edges = [(index[u], index[v]) for u, v in graph.edges()]
    fd_in, in_file = tempfile.mkstemp()
    fd_out, out_file = tempfile.mkstemp()
    try:
        with os.fdopen(fd_in, "w") as f:
            f.write(f"{graph.number_of_nodes()} {len(edges)}\n")
            f.writelines(f"{u} {v}\n" for u, v in edges)
        proc = subprocess.run(
            [orca_path, "node", "4", in_file, out_file],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ORCA failed: {proc.stderr.strip()}")
        return np.loadtxt(out_file, ndmin=2)
    finally:
        os.close(fd_out)
        for path in (in_file, out_file):
            try:
                os.remove(path)
            except OSError:
                pass


def _spearman_matrix(x: np.ndarray) -> np.ndarray:
    ranks = np.apply_along_axis(rankdata, 1, x).astype(float)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    centered /= np.sqrt((centered * centered).sum(axis=1, keepdims=True) + 1e-8)
    corr = centered @ centered.T
    np.clip(corr, -1.0, 1.0, out=corr)
    return corr


def gcd_embedding(graph: nx.Graph, orca_path: str = DEFAULT_ORCA_PATH) -> np.ndarray:
    gcm = _spearman_matrix(_orbit_counts(graph, orca_path)[:, _ORBITS].T)
    np.nan_to_num(gcm, copy=False)
    return gcm[np.triu_indices_from(gcm, k=1)]


def gcd_embeddings(
    graphs: list[nx.Graph], orca_path: str = DEFAULT_ORCA_PATH, n_jobs: int = -1
) -> list[np.ndarray]:
    return Parallel(n_jobs=n_jobs, batch_size=1)(delayed(gcd_embedding)(g, orca_path) for g in graphs)
