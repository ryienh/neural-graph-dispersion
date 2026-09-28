"""Two-phase selection of a diverse set of graphs from edge-probability matrices."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from tqdm import tqdm

from ngd.config import NGDConfig
from ngd.metrics import diversity, embed
from ngd.metrics.portrait import js_divergence, portrait_distributions, portraits
from ngd.sampling import adj_to_nx, sample, sample_bernoulli


@dataclass
class SelectionResult:
    graphs: list[nx.Graph]
    diversity: float
    temperatures: list[float]
    sources: list[int]


def select(probs: np.ndarray, metric: str, config: NGDConfig) -> SelectionResult:
    """Select ``config.k_select`` graphs from ``probs`` (shape (H, N, N)) for ``metric``."""
    rng = np.random.default_rng(config.seed)
    run = _select_chunked if config.memory_efficient else _select_in_memory
    graphs, temperatures, sources = run(probs, metric, config, rng)
    score = diversity(graphs, metric, config.orca_path, n_jobs=config.n_jobs)
    return SelectionResult(graphs, score, temperatures, sources)


def _select_in_memory(probs, metric, config, rng):
    n_phase1, n_top, per_matrix = _budgets(len(probs), config)
    quiet = not config.verbose

    graphs = [
        adj_to_nx(sample_bernoulli(probs[i]))
        for i in tqdm(range(n_phase1), desc="  Phase 1 sampling", disable=quiet)
    ]
    reps = _representations(graphs, metric, config)
    top = _disperse(reps, metric, n_top)

    pool = [graphs[i] for i in top]
    pool_reps = [reps[i] for i in top]
    temperatures = [1.0] * len(top)
    sources = list(top)
    new = []
    for m in tqdm(top, desc="  Phase 2 sampling", disable=quiet):
        for _ in range(per_matrix - 1):
            temp = _temperature(rng, config)
            new.append(adj_to_nx(sample(probs[m], temp)))
            temperatures.append(temp)
            sources.append(m)
    if new:
        pool += new
        pool_reps += _representations(new, metric, config)

    final = _disperse(pool_reps, metric, config.k_select)
    return [pool[i] for i in final], [temperatures[i] for i in final], [sources[i] for i in final]


def _select_chunked(probs, metric, config, rng):
    n_phase1, n_top, per_matrix = _budgets(len(probs), config)

    phase1 = [(i, 1.0, _seed(rng)) for i in range(n_phase1)]
    top = _disperse(_chunked_representations(probs, phase1, metric, config), metric, n_top)

    pool = []
    for m in (phase1[i][0] for i in top):
        for _ in range(per_matrix):
            seed = _seed(rng)
            pool.append((m, _temperature(rng, config), seed))
    final = _disperse(_chunked_representations(probs, pool, metric, config), metric, config.k_select)

    chosen = [pool[i] for i in final]
    graphs = [adj_to_nx(_draw(probs, c)) for c in chosen]
    return graphs, [t for _, t, _ in chosen], [m for m, _, _ in chosen]


def _budgets(n_matrices: int, config: NGDConfig) -> tuple[int, int, int]:
    n_phase1 = min(n_matrices, config.sampling_budget // 10)
    n_top = min(config.n_top_matrices, n_matrices, n_phase1)
    per_matrix = (config.sampling_budget - n_phase1) // n_top if n_top > 0 else 0
    return n_phase1, n_top, per_matrix


def _seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31))


def _temperature(rng: np.random.Generator, config: NGDConfig) -> float:
    if rng.random() < config.rigid_prob:
        return np.inf
    lo, hi = np.log(config.temp_range[0]), np.log(config.temp_range[1])
    return float(np.exp(rng.uniform(lo, hi)))


def _draw(probs: np.ndarray, candidate: tuple[int, float, int]) -> np.ndarray:
    m, temp, seed = candidate
    return sample(probs[m], temp, np.random.RandomState(seed))


def _representations(graphs: list[nx.Graph], metric: str, config: NGDConfig) -> list[np.ndarray]:
    if metric == "portrait_div":
        return portraits(graphs, n_jobs=config.n_jobs)
    return list(embed(graphs, metric, config.orca_path, n_jobs=config.n_jobs))


def _chunked_representations(probs, candidates, metric, config) -> list[np.ndarray]:
    reps = []
    starts = range(0, len(candidates), config.chunk_size)
    for start in tqdm(starts, desc="  Sampling chunks", disable=not config.verbose):
        chunk = candidates[start : start + config.chunk_size]
        reps += _representations([adj_to_nx(_draw(probs, c)) for c in chunk], metric, config)
    return reps


class _Geometry:
    def __init__(self, reps: list[np.ndarray], metric: str):
        self.portrait = metric == "portrait_div"
        self.x = portrait_distributions(reps) if self.portrait else np.array(reps)

    def __len__(self) -> int:
        return len(self.x)

    def distances_from(self, i: int, candidates: np.ndarray) -> np.ndarray:
        if self.portrait:
            return np.array([js_divergence(self.x[c], self.x[i]) for c in candidates])
        return np.linalg.norm(self.x - self.x[i], axis=1)[candidates]

    def dissimilar_pair(self) -> tuple[int, int]:
        if self.portrait:
            if len(self) <= 2:
                return 0, min(1, len(self) - 1)
            a = int(np.argmax([js_divergence(self.x[0], v) for v in self.x]))
            b = int(np.argmax([js_divergence(self.x[a], v) for v in self.x]))
            return a, b
        a = int(np.argmax(np.linalg.norm(self.x - self.x.mean(axis=0), axis=1)))
        b = int(np.argmax(np.linalg.norm(self.x - self.x[a], axis=1)))
        return a, b


def _disperse(reps: list[np.ndarray], metric: str, k: int) -> list[int]:
    geometry = _Geometry(reps, metric)
    return greedy_maxsum(geometry, k, start=geometry.dissimilar_pair())


def greedy_maxsum(geometry: _Geometry, k: int, start: tuple[int, ...]) -> list[int]:
    n = len(geometry)
    if k > n:
        raise ValueError(f"Cannot select {k} graphs from {n} candidates.")
    scores = np.zeros(n)
    available = np.ones(n, dtype=bool)
    selected: list[int] = []

    def add(i: int) -> None:
        selected.append(i)
        available[i] = False
        candidates = np.flatnonzero(available)
        scores[candidates] += geometry.distances_from(i, candidates)

    for i in start:
        add(i)
    scores[~available] = -np.inf
    while len(selected) < k:
        best = int(np.argmax(scores))
        if scores[best] == -np.inf:
            break
        scores[best] = -np.inf
        add(best)
    return selected[:k]
