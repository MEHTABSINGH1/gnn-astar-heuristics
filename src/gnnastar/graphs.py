"""Download, clean and cache city road networks as compact CSR graphs.

Each city is the drivable OpenStreetMap network within a fixed radius of its
centre, made undirected and reduced to its largest connected component. We
cache only what the experiments need (node coordinates and edge lengths) in
``data/graphs/<city>.npz`` so later runs never touch the network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "graphs"
EARTH_R = 6_371_008.8  # mean Earth radius in metres

# name -> (lat, lon, radius_m, region). Centres are approximate city centres.
CITIES: dict[str, tuple[float, float, int, str]] = {
    "chandigarh": (30.7333, 76.7794, 5000, "Punjab/Haryana"),
    "ludhiana":   (30.9010, 75.8573, 5000, "Punjab"),
    "jalandhar":  (31.3260, 75.5762, 5000, "Punjab"),
    "patiala":    (30.3398, 76.3869, 5000, "Punjab"),
    "amritsar":   (31.6340, 74.8723, 5000, "Punjab"),
    "dharwad":    (15.4589, 75.0078, 5000, "Karnataka"),
    "mysuru":     (12.2958, 76.6394, 5000, "Karnataka"),
    "bengaluru":  (12.9716, 77.5946, 5000, "Karnataka"),
}


@dataclass
class RoadGraph:
    name: str
    lat: np.ndarray      # degrees, shape (n,)
    lon: np.ndarray      # degrees, shape (n,)
    indptr: np.ndarray   # CSR row pointers, shape (n+1,)
    indices: np.ndarray  # CSR neighbours
    weights: np.ndarray  # edge length in metres (symmetric graph)

    @property
    def n(self) -> int:
        return len(self.lat)

    @property
    def m(self) -> int:  # undirected edges
        return len(self.indices) // 2

    def csr(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.weights, self.indices, self.indptr), shape=(self.n, self.n))

    def edge_index(self) -> np.ndarray:
        rows = np.repeat(np.arange(self.n), np.diff(self.indptr))
        return np.stack([rows, self.indices])

    def haversine_to(self, t: int) -> np.ndarray:
        """Great-circle distance (m) from every node to node t: an admissible heuristic,
        because every road edge is at least as long as the straight line between its ends."""
        return haversine(self.lat, self.lon, self.lat[t], self.lon[t])


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def download(name: str) -> Path:
    """Fetch one city from OpenStreetMap (needs internet) and cache it."""
    import osmnx as ox

    lat, lon, dist, _ = CITIES[name]
    G = ox.graph_from_point((lat, lon), dist=dist, network_type="drive", simplify=True)
    G = ox.truncate.largest_component(G, strongly=False)  # needs the directed graph
    G = ox.convert.to_undirected(G)
    nodes = list(G.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    la = np.array([G.nodes[v]["y"] for v in nodes])
    lo = np.array([G.nodes[v]["x"] for v in nodes])
    # keep the shortest of any parallel edges, store both directions
    best: dict[tuple[int, int], float] = {}
    for u, v, d in G.edges(data=True):
        a, b = idx[u], idx[v]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        best[key] = min(best.get(key, np.inf), float(d["length"]))
    r = np.array([k[0] for k in best] + [k[1] for k in best])
    c = np.array([k[1] for k in best] + [k[0] for k in best])
    w = np.array(list(best.values()) * 2)
    # an edge can never be shorter than the straight line between its ends
    w = np.maximum(w, haversine(la[r], lo[r], la[c], lo[c]))
    A = sp.csr_matrix((w, (r, c)), shape=(len(nodes), len(nodes)))
    A.sort_indices()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{name}.npz"
    np.savez_compressed(out, lat=la, lon=lo, indptr=A.indptr, indices=A.indices, weights=A.data)
    return out


def load(name: str) -> RoadGraph:
    f = DATA_DIR / f"{name}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run `python scripts/download.py {name}` first")
    z = np.load(f)
    return RoadGraph(name, z["lat"], z["lon"], z["indptr"], z["indices"], z["weights"])
