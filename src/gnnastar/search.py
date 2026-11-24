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
    expanded: list | None = None   # expansion order (only with record=True)
    path: list | None = None       # s ... t (only with record=True)


def astar(g: RoadGraph, s: int, t: int, h: np.ndarray | None, record: bool = False) -> SearchResult:
    """A* from s to t. ``h`` is a per-node heuristic array for target t (None = Dijkstra).
    With ``record=True`` the expanded nodes and the returned path are kept (for plots)."""
    indptr, indices, weights = g.indptr, g.indices, g.weights
    hv = h if h is not None else np.zeros(g.n)
    gcost = np.full(g.n, np.inf)
    closed = np.zeros(g.n, dtype=bool)
    gcost[s] = 0.0
    heap = [(hv[s], 0.0, s)]
    expansions = 0
    order, parent = ([], np.full(g.n, -1)) if record else (None, None)
    while heap:
        f, gu, u = heapq.heappop(heap)
        if gu > gcost[u]:
            continue  # stale entry
        if u == t:
            path = None
            if record:
                path, x = [t], t
                while x != s:
                    x = int(parent[x]); path.append(x)
                path.reverse()
            return SearchResult(gu, expansions, order, path)
        if closed[u]:
            continue
        closed[u] = True
        expansions += 1
        if record:
            order.append(u)
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            nv = gu + weights[k]
            if nv < gcost[v]:
                gcost[v] = nv
                closed[v] = False  # re-open if needed (inconsistent heuristics)
                if record:
                    parent[v] = u
                heapq.heappush(heap, (nv + hv[v], nv, v))
    return SearchResult(np.inf, expansions, order, None)


class ALT:
    """A*, Landmarks and Triangle inequality (Goldberg & Harrelson, 2005).

    Landmarks are picked by farthest-point selection; for an undirected graph
    h(v) = max_l |d(l, t) - d(l, v)| is admissible and consistent.
    """

    def __init__(self, g: RoadGraph, n_landmarks: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        A = g.csr()
        self.directed = g.directed
        # farthest-point selection, started from the node farthest from a random node.
        # Landmark k is chosen given landmarks 1..k-1, so the first k rows are exactly
        # ALT with k landmarks (a budget of k single-source Dijkstras; 2k when directed,
        # because distances to AND from each landmark are needed).
        def from_(l):  # d(l, v) for all v
            return sp_dijkstra(A, directed=self.directed, indices=l)

        AT = A.T.tocsr() if self.directed else None

        def to_(l):    # d(v, l) for all v: Dijkstra on the reversed graph
            return sp_dijkstra(AT, directed=True, indices=l)

        far = from_(int(rng.integers(g.n)))
        lms = [int(np.argmax(far))]
        Df = np.atleast_2d(from_(lms))
        Dt = np.atleast_2d(to_(lms)) if self.directed else Df
        while len(lms) < n_landmarks:
            nxt = int(np.argmax(np.minimum(Df, Dt).min(axis=0)))
            lms.append(nxt)
            Df = np.vstack([Df, from_(nxt)[None, :]])
            Dt = np.vstack([Dt, to_(nxt)[None, :]]) if self.directed else Df
        self.landmarks, self.Df, self.Dt = lms, Df, Dt

    def heuristic(self, t: int, k: int | None = None) -> np.ndarray:
        Df = self.Df if k is None else self.Df[:k]
        Dt = self.Dt if k is None else self.Dt[:k]
        # triangle inequality: d(v,t) >= d(v,l) - d(t,l)  and  d(v,t) >= d(l,t) - d(l,v)
        h = np.maximum(Dt - Dt[:, t][:, None], Df[:, t][:, None] - Df)
        return np.maximum(h.max(axis=0), 0.0)


def distances_to(g: RoadGraph, targets) -> np.ndarray:
    """Exact cost from every node TO each target (rows = targets)."""
    A = g.csr()
    if g.directed:
        return sp_dijkstra(A.T.tocsr(), directed=True, indices=np.atleast_1d(targets))
    return sp_dijkstra(A, directed=False, indices=np.atleast_1d(targets))
