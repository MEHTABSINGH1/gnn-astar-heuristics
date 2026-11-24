"""Learned A* heuristic: predict the road/straight-line detour ratio.

For a node v and target t we predict r(v, t) = d_road(v, t) / d_haversine(v, t) >= 1
and use h(v) = d_haversine(v, t) * r_hat(v, t). Predicting the *ratio* instead
of the raw distance makes the target scale-free, so a model trained on one
city transfers to another.

Encoders (the ablation):
* ``mlp``  - node features only, no message passing (tests whether graph context matters)
* ``sage`` - GraphSAGE (Hamilton et al., 2017)
* ``gat``  - GAT with edge length as an edge feature (Velickovic et al., 2018)

Training uses the pinball (quantile) loss on log r. A low quantile tau makes the
model under-estimate on purpose, trading fewer A* expansions for fewer
sub-optimal paths; tau is the knob we sweep.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, SAGEConv

from .graphs import RoadGraph


def node_features(g: RoadGraph) -> np.ndarray:
    """City-invariant node features (no absolute coordinates).

    dist graphs: 6 features (position relative to centroid, degree, mean edge length, dead end, junction).
    time graphs: the same 6 plus 4 road-class features (mean / max incident speed, share of fast
    roads, one-way imbalance), so the model can tell arterials from residential lanes."""
    out_deg = np.diff(g.indptr).astype(float)
    rows = np.repeat(np.arange(g.n), np.diff(g.indptr))
    lengths = g.lengths if g.lengths is not None else g.weights
    mean_len = np.bincount(rows, weights=lengths, minlength=g.n) / np.maximum(out_deg, 1)
    deg = out_deg
    if g.directed:
        in_deg = np.bincount(g.indices, minlength=g.n).astype(float)
        deg = np.maximum(out_deg, in_deg)
    # local coordinates in km relative to the city centroid (translation-invariant)
    lat0, lon0 = g.lat.mean(), g.lon.mean()
    y = (g.lat - lat0) * 111.32
    x = (g.lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    feats = [x / 5, y / 5, np.log1p(deg), np.log1p(mean_len) / 5,
             (deg == 1).astype(float), (deg >= 4).astype(float)]
    if g.mode == "time":
        kph = lengths / g.weights * 3.6
        cnt = np.maximum(np.bincount(rows, minlength=g.n), 1)
        mean_kph = np.bincount(rows, weights=kph, minlength=g.n) / cnt
        max_kph = np.zeros(g.n); np.maximum.at(max_kph, rows, kph)
        fast = np.bincount(rows, weights=(kph >= 40).astype(float), minlength=g.n) / cnt
        feats += [mean_kph / 40, max_kph / 40, fast, (out_deg - in_deg) / np.maximum(deg, 1)]
    return np.stack(feats, axis=1).astype(np.float32)


def pair_features(g: RoadGraph, v: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Geometry of the (v, t) pair: straight-line distance and direction."""
    d = np.maximum(_hav(g, v, t), 1.0)
    dy = (g.lat[t] - g.lat[v]) * 111_320
    dx = (g.lon[t] - g.lon[v]) * 111_320 * np.cos(np.radians(g.lat[v]))
    cols = [np.log(d) / 10, dx / d, dy / d]
    if g.mode == "time":  # the base heuristic divides by this city's top speed, so the model must know it
        cols.append(np.full(len(d), np.log(3.6 / g.scale / 60)))
    return np.stack(cols, axis=1).astype(np.float32)


def _hav(g: RoadGraph, v, t):
    from .graphs import haversine
    return haversine(g.lat[v], g.lon[v], g.lat[t], g.lon[t])


class HeuristicNet(nn.Module):
    def __init__(self, encoder: str = "sage", in_dim: int = 6, hidden: int = 64, layers: int = 3,
                 pair_dim: int = 3):
        super().__init__()
        self.encoder = encoder
        self.inp = nn.Linear(in_dim, hidden)
        if encoder == "sage":
            self.convs = nn.ModuleList(SAGEConv(hidden, hidden) for _ in range(layers))
        elif encoder == "gat":
            self.convs = nn.ModuleList(GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=1)
                                       for _ in range(layers))
        elif encoder == "mlp":
            self.convs = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        else:
            raise ValueError(encoder)
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(4 * hidden + pair_dim, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def embed(self, x, edge_index, edge_len):
        z = torch.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms):
            if self.encoder == "sage":
                upd = conv(z, edge_index)
            elif self.encoder == "gat":
                upd = conv(z, edge_index, edge_attr=edge_len)
            else:
                upd = conv(z)
            z = norm(z + torch.relu(upd))  # residual keeps deep stacks stable
        return z

    def log_ratio(self, z, v, t, pf):
        """Predicted log(d_road / d_hav) >= 0 for node pairs (v, t)."""
        zv, zt = z[v], z[t]
        out = self.head(torch.cat([zv, zt, zv * zt, (zv - zt).abs(), pf], dim=1)).squeeze(1)
        return nn.functional.softplus(out)


def graph_tensors(g: RoadGraph, device):
    """Node features and message-passing edges. Messages flow both ways along every road,
    even one-way streets, because a node's context does not depend on driving direction."""
    x = torch.tensor(node_features(g), device=device)
    ei, w = g.edge_index(), g.weights
    if g.directed:
        ei, w = np.concatenate([ei, ei[::-1]], axis=1), np.r_[w, w]
    el = torch.tensor(np.log1p(w)[:, None] / 5, dtype=torch.float32, device=device)
    return x, torch.tensor(ei, dtype=torch.long, device=device), el


def pinball(pred, target, tau: float):
    diff = target - pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))


@torch.no_grad()
def heuristic_for_target(model: HeuristicNet, g: RoadGraph, z: torch.Tensor, t: int) -> np.ndarray:
    """h(v) for all v towards target t, in metres (one batched forward pass)."""
    v = np.arange(g.n)
    tt = np.full(g.n, t)
    pf = torch.tensor(pair_features(g, v, tt), device=z.device)
    r = model.log_ratio(z, torch.as_tensor(v, device=z.device), torch.as_tensor(tt, device=z.device), pf)
    h = g.base_heuristic(t) * np.exp(r.cpu().numpy())
    h[t] = 0.0
    return h
