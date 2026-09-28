"""End-to-end NGD: train the generator ensemble, then select a diverse set per metric."""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from ngd.config import NGDConfig
from ngd.features import SpectralProjection
from ngd.metrics import netlsd
from ngd.selection import SelectionResult, select
from ngd.train import train_ensemble


def run_pipeline(config: NGDConfig) -> dict[str, SelectionResult]:
    """Run NGD and return the selected graphs for each metric in ``config.metrics``."""
    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
    netlsd.set_device(config.device)

    projection = SpectralProjection(
        config.features, config.num_ensembles, config.projection_dim, config.direction_seed
    )
    probs = train_ensemble(config, projection)

    results = {}
    for metric in config.metrics:
        results[metric] = select(probs, metric, config)
        if config.verbose:
            print(f"{metric}: diversity {results[metric].diversity:.6f}")

    if config.output_dir is not None:
        save_results(results, config)
    return results


def save_results(results: dict[str, SelectionResult], config: NGDConfig) -> None:
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for metric, result in results.items():
        with open(out / f"graphs_{metric}.pkl", "wb") as f:
            pickle.dump(result.graphs, f)
    summary = {
        "diversity": {metric: result.diversity for metric, result in results.items()},
        "config": dataclasses.asdict(config),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
