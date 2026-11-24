"""Correctness checks on a synthetic grid graph (no downloads needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnnastar.graphs import RoadGraph, haversine  # noqa: E402
from gnnastar.search import ALT, astar, distances_to  # noqa: E402


def grid(k=15, seed=0) -> RoadGraph:
    """k x k street grid around (30.7N, 76.8E), ~100 m blocks, random detours on edges."""
    rng = np.random.default_rng(seed)
    lat = 30.7 + np.repeat(np.arange(k), k) * 0.0009
    lon = 76.8 + np.tile(np.arange(k), k) * 0.00105
    r, c = [], []
    for i in range(k):
        for j in range(k):
            u = i * k + j
            if j + 1 < k: r.append(u); c.append(u + 1)
            if i + 1 < k: r.append(u); c.append(u + k)
    r, c = np.array(r), np.array(c)
    w = haversine(lat[r], lon[r], lat[c], lon[c]) * rng.uniform(1.0, 1.6, len(r))
    A = sp.csr_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(k * k, k * k))
    A.sort_indices()
    return RoadGraph("grid", lat, lon, A.indptr, A.indices, A.data)


@pytest.fixture(scope="module")
def g():
    return grid()


def test_haversine_is_admissible(g):
    D = distances_to(g, [0, 50, 224])
    for row, t in zip(D, [0, 50, 224]):
        assert np.all(g.haversine_to(t) <= row + 1e-6)


def test_all_exact_methods_agree_with_dijkstra(g):
    alt = ALT(g, n_landmarks=4)
    rng = np.random.default_rng(1)
    for _ in range(30):
        s, t = (int(x) for x in rng.integers(g.n, size=2))
        exact = distances_to(g, t)[0, s]
        for h in (None, g.haversine_to(t), alt.heuristic(t)):
            assert astar(g, s, t, h).cost == pytest.approx(exact)


def test_informed_search_expands_fewer_nodes(g):
    alt = ALT(g, n_landmarks=4)
    s, t = 0, g.n - 1
    e_dij = astar(g, s, t, None).expansions
    e_hav = astar(g, s, t, g.haversine_to(t)).expansions
    e_alt = astar(g, s, t, alt.heuristic(t)).expansions
    assert e_hav < e_dij and e_alt <= e_dij


def test_overestimating_heuristic_still_finds_a_path(g):
    s, t = 3, 200
    h = g.haversine_to(t) * 3.0           # deliberately inadmissible
    exact = distances_to(g, t)[0, s]
    r = astar(g, s, t, h)
    assert np.isfinite(r.cost) and r.cost >= exact - 1e-6


def test_model_forward_and_heuristic_shape(g):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from gnnastar.model import HeuristicNet, graph_tensors, heuristic_for_target
    for enc in ("mlp", "sage", "gat"):
        m = HeuristicNet(enc).eval()
        z = m.embed(*graph_tensors(g, "cpu"))
        h = heuristic_for_target(m, g, z, 17)
        assert h.shape == (g.n,) and h[17] == 0 and np.all(h >= g.haversine_to(17) - 1e-6)


def directed_grid(k=14, seed=3) -> RoadGraph:
    """Grid with travel-time costs, random speeds and ~25% one-way streets (still strongly connected:
    every street has a two-way twin one block away)."""
    rng = np.random.default_rng(seed)
    base = grid(k, seed)
    A = base.csr().tocoo()
    keep = np.ones(len(A.row), bool)
    for e, (u, v) in enumerate(zip(A.row, A.col)):
        i, j = divmod(u, k)
        # make horizontal streets in odd rows (except the border) one-way eastbound
        if i % 2 == 1 and 0 < i < k - 1 and v == u - 1:
            keep[e] = False
    r, c, length = A.row[keep], A.col[keep], A.data[keep]
    kph = rng.choice([20, 30, 50], size=len(r))
    cost = length / (kph / 3.6)
    M = sp.csr_matrix((cost, (r, c)), shape=(k * k, k * k)); M.sort_indices()
    L = sp.csr_matrix((length, (r, c)), shape=(k * k, k * k)); L.sort_indices()
    return RoadGraph("dgrid", base.lat, base.lon, M.indptr, M.indices, M.data, directed=True,
                     scale=1 / (50 / 3.6), lengths=L.data, mode="time")


def test_directed_travel_time_methods_are_exact():
    g = directed_grid()
    alt = ALT(g, n_landmarks=4)
    rng = np.random.default_rng(2)
    for _ in range(30):
        s, t = (int(x) for x in rng.integers(g.n, size=2))
        exact = distances_to(g, t)[0, s]
        assert np.isfinite(exact)
        assert np.all(g.base_heuristic(t) <= distances_to(g, t)[0] + 1e-6)   # admissible
        assert np.all(alt.heuristic(t) <= distances_to(g, t)[0] + 1e-6)      # admissible
        for h in (None, g.base_heuristic(t), alt.heuristic(t)):
            assert astar(g, s, t, h).cost == pytest.approx(exact)


def test_model_runs_on_directed_graph():
    pytest.importorskip("torch_geometric")
    from gnnastar.model import HeuristicNet, graph_tensors, heuristic_for_target, node_features, pair_features
    g = directed_grid()
    m = HeuristicNet("sage", in_dim=node_features(g).shape[1],
                     pair_dim=pair_features(g, np.array([0]), np.array([1])).shape[1]).eval()
    z = m.embed(*graph_tensors(g, "cpu"))
    h = heuristic_for_target(m, g, z, 5)
    assert h.shape == (g.n,) and h[5] == 0 and np.all(h >= g.base_heuristic(5) - 1e-9)
