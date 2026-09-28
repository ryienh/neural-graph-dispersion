"""NetLSD heat and wave trace signatures (Tsitsulin et al., 2018), computed in PyTorch."""

from __future__ import annotations

import netlsd
import networkx as nx
import numpy as np
import torch

_device: torch.device | None = None
_timescales: dict[tuple[str, torch.device], torch.Tensor] = {}


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def set_device(device: str | torch.device) -> None:
    global _device
    _device = torch.device(device)


def _timescale(kernel: str, device: torch.device) -> torch.Tensor:
    key = (kernel, device)
    if key not in _timescales:
        if kernel == "heat":
            t = torch.logspace(-2, 2, 250, device=device)
        else:
            t = torch.linspace(0, 2.0 * np.pi, 250, device=device)
        _timescales[key] = t.unsqueeze(1)
    return _timescales[key]


def _normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1)
    inv_sqrt = torch.zeros_like(deg)
    nonzero = deg > 0
    inv_sqrt[nonzero] = 1.0 / torch.sqrt(deg[nonzero])
    d = torch.diag(inv_sqrt)
    return d @ (torch.diag(deg) - adj) @ d


def signature(graph: nx.Graph, kernel: str) -> np.ndarray:
    device = get_device()
    adj = torch.from_numpy(nx.to_numpy_array(graph, dtype=np.float32)).to(device)
    try:
        eigvals = torch.linalg.eigvalsh(_normalized_laplacian(adj)).unsqueeze(0)
        t = _timescale(kernel, device)
        if kernel == "heat":
            trace = torch.exp(-t * eigvals).sum(dim=1)
        else:
            trace = torch.exp(-1j * t * eigvals).sum(dim=1).real
        return (trace / adj.shape[0]).cpu().numpy()
    except torch.linalg.LinAlgError:
        fn = netlsd.heat if kernel == "heat" else netlsd.wave
        return np.asarray(fn(graph), dtype=np.float32)


def signatures(graphs: list[nx.Graph], kernel: str) -> list[np.ndarray]:
    return [signature(g, kernel) for g in graphs]
