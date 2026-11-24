"""Shortest-path search with pluggable heuristics.

All methods share one A* implementation, so differences in node expansions
come only from the heuristic:

* ``dijkstra``   h = 0                                  (exact)
* ``haversine``  h = great-circle distance to target    (admissible, exact)
* ``alt``        landmark triangle-inequality bound     (admissible, exact)
* ``learned``    h = haversine x predicted detour ratio (may overestimate,
                 so the returned path can be sub-optimal; we measure how much)

The implementation re-opens closed nodes when a cheaper path is found, so it
stays correct for inconsistent heuristics and only sub-optimality due to
over-estimation remains.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

from .graphs import RoadGraph


@dataclass
class SearchResult:
    cost: float
    expansions: int


def astar(g: RoadGraph, s: int, t: int, h: np.ndarray | None) -> SearchResult:
    """A* from s to t. ``h`` is a per-node heuristic array for target t (None = Dijkstra)."""
    indptr, indices, weights = g.indptr, g.indices, g.weights
    hv = h if h is not None else np.zeros(g.n)
    gcost = np.full(g.n, np.inf)
    closed = np.zeros(g.n, dtype=bool)
    gcost[s] = 0.0
    heap = [(hv[s], 0.0, s)]
    expansions = 0
    while heap:
        f, gu, u = heapq.heappop(heap)
        if gu > gcost[u]:
            continue  # stale entry
        if u == t:
            return SearchResult(gu, expansions)
        if closed[u]:
            continue
        closed[u] = True
        expansions += 1
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            nv = gu + weights[k]
            if nv < gcost[v]:
                gcost[v] = nv
                closed[v] = False  # re-open if needed (inconsistent heuristics)
                heapq.heappush(heap, (nv + hv[v], nv, v))
    return SearchResult(np.inf, expansions)


class ALT:
    """A*, Landmarks and Triangle inequality (Goldberg & Harrelson, 2005).

    Landmarks are picked by farthest-point selection; for an undirected graph
    h(v) = max_l |d(l, t) - d(l, v)| is admissible and consistent.
    """

    def __init__(self, g: RoadGraph, n_landmarks: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        A = g.csr()
        # farthest-point selection, started from the node farthest from a random node.
        # Landmark k is chosen given landmarks 1..k-1, so the first k rows are exactly
        # ALT with k landmarks (a budget of k single-source Dijkstras).
        far = sp_dijkstra(A, directed=False, indices=int(rng.integers(g.n)))
        lms = [int(np.argmax(far))]
        D = sp_dijkstra(A, directed=False, indices=lms)
        while len(lms) < n_landmarks:
            nxt = int(np.argmax(D.min(axis=0)))
            lms.append(nxt)
            D = np.vstack([D, sp_dijkstra(A, directed=False, indices=nxt)[None, :]])
        self.landmarks, self.D = lms, D

    def heuristic(self, t: int, k: int | None = None) -> np.ndarray:
        D = self.D if k is None else self.D[:k]
        return np.max(np.abs(D[:, t][:, None] - D), axis=0)


def distances_to(g: RoadGraph, targets) -> np.ndarray:
    """Exact road distance from every node to each target (rows = targets)."""
    return sp_dijkstra(g.csr(), directed=False, indices=np.atleast_1d(targets))
