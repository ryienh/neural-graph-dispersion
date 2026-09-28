"""Training of the generator ensemble under the Coulomb dispersion loss."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ngd.config import NGDConfig
from ngd.features import SpectralProjection, coulomb_loss
from ngd.model import EnsembleGraphGenerator


def train_ensemble(config: NGDConfig, projection: SpectralProjection) -> np.ndarray:
    """Train the ensemble and collect edge-probability snapshots along its trajectory."""
    device = config.device
    model = EnsembleGraphGenerator(
        num_ensembles=config.num_ensembles,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        num_hidden=config.num_hidden,
        n_vertices=config.n_vertices,
        num_particles=config.batch_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    z = torch.randn(config.batch_size, config.latent_dim, device=device)
    particle_ids = torch.arange(config.batch_size, device=device)
    projection.to(device)

    n, k = config.n_vertices, config.graphs_per_ensemble
    probs = np.empty((config.num_ensembles, k, n, n), dtype=np.float32)
    collected = 0
    snapshots = config.snapshot_iterations

    pbar = tqdm(range(config.num_iterations), desc="Training", disable=not config.verbose)
    for iteration in pbar:
        optimizer.zero_grad()
        p = model(z, particle_ids)
        p_rep = p.unsqueeze(0).expand(config.num_samples_ste, -1, -1, -1, -1)
        p_ste = p_rep + (torch.bernoulli(p_rep) - p_rep).detach()

        losses = [
            coulomb_loss(
                projection(p_ste[:, e].reshape(-1, n, n), ensemble_id=e),
                gamma=config.gamma,
                epsilon=config.epsilon,
            )
            for e in range(config.num_ensembles)
        ]
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()

        # Unused draw; keeps seeded runs identical to the paper's.
        torch.bernoulli(p.detach())
        if iteration in snapshots and collected < k:
            take = min(config.batch_size, k - collected)
            probs[:, collected : collected + take] = p.detach()[:, :take].cpu().numpy()
            collected += take

        pbar.set_postfix(loss=f"{loss.item() / config.num_ensembles:.4f}")

    return probs.reshape(-1, n, n)
