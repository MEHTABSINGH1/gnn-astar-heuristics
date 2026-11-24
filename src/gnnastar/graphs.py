"""Download, clean and cache city road networks as compact CSR graphs.

Two graph modes:

* ``dist`` - undirected, edge weight = length in metres (study 1).
* ``time`` - directed (one-way streets respected), edge weight = free-flow
  travel time in seconds from OSM ``maxspeed`` tags, with road-class defaults
  where the tag is missing (study 2, closer to real navigation).

Each city is the drivable OpenStreetMap network within a fixed radius of its
centre, reduced to its largest (weakly / strongly) connected component. Only
what the experiments need is cached in ``data/graphs[_time]/<city>.npz``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "graphs"
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
    # city-scale test graphs (scaling study)
    "bengaluru_xl": (12.9716, 77.5946, 12000, "Karnataka"),
    "ludhiana_xl":  (30.9010, 75.8573, 10000, "Punjab"),
}

# free-flow speeds (km/h) used when OSM has no maxspeed tag: conservative Indian urban values
DEFAULT_KPH = {"motorway": 80, "trunk": 60, "primary": 50, "secondary": 40, "tertiary": 35,
               "unclassified": 25, "residential": 20, "living_street": 10, "service": 15}


def data_dir(mode: str) -> Path:
    return DATA_DIR if mode == "dist" else ROOT / "data" / f"graphs_{mode}"


@dataclass
class RoadGraph:
    name: str
    lat: np.ndarray      # degrees, shape (n,)
    lon: np.ndarray      # degrees, shape (n,)
    indptr: np.ndarray   # CSR row pointers (out-edges), shape (n+1,)
    indices: np.ndarray  # CSR edge heads
    weights: np.ndarray  # edge cost: metres (dist) or seconds (time)
    directed: bool = False
    scale: float = 1.0   # base heuristic = haversine * scale (1 for dist, 1/v_max for time)
    lengths: np.ndarray | None = None  # metres per edge (time mode)
    mode: str = "dist"

    @property
    def n(self) -> int:
        return len(self.lat)

    @property
    def m(self) -> int:
        return len(self.indices) if self.directed else len(self.indices) // 2

    def csr(self) -> sp.csr_matrix:
        return sp.csr_matrix((self.weights, self.indices, self.indptr), shape=(self.n, self.n))

    def edge_index(self) -> np.ndarray:
        rows = np.repeat(np.arange(self.n), np.diff(self.indptr))
        return np.stack([rows, self.indices])

    def haversine_to(self, t: int) -> np.ndarray:
        """Great-circle distance (m) from every node to node t."""
        return haversine(self.lat, self.lon, self.lat[t], self.lon[t])

    def base_heuristic(self, t: int) -> np.ndarray:
        """Admissible straight-line bound in the graph's cost unit.

        dist: every edge is at least as long as the straight line between its ends.
        time: every edge takes at least length / v_max, where v_max is the fastest
        edge speed in the graph, so haversine / v_max never over-estimates."""
        return self.haversine_to(t) * self.scale


def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _edge_kph(d: dict) -> float:
    ms = d.get("speed_kph")
    if ms is not None and np.isfinite(ms) and ms > 0:
        return float(ms)
    hw = d.get("highway", "unclassified")
    hw = hw[0] if isinstance(hw, list) else hw
    return float(DEFAULT_KPH.get(str(hw).replace("_link", ""), 25))


def download(name: str, mode: str = "dist") -> Path:
    """Fetch one city from OpenStreetMap (needs internet; responses are cached) and save it."""
    import osmnx as ox

    lat, lon, dist, _ = CITIES[name]
    G = ox.graph_from_point((lat, lon), dist=dist, network_type="drive", simplify=True)
    if mode == "time":
        G = ox.truncate.largest_component(G, strongly=True)    # every node reachable both ways
        G = ox.routing.add_edge_speeds(G, hwy_speeds=DEFAULT_KPH)
    else:
        G = ox.truncate.largest_component(G, strongly=False)   # needs the directed graph
        G = ox.convert.to_undirected(G)
    nodes = list(G.nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    la = np.array([G.nodes[v]["y"] for v in nodes])
    lo = np.array([G.nodes[v]["x"] for v in nodes])

    best: dict[tuple[int, int], tuple[float, float]] = {}   # (u,v) -> (cost, length), cheapest parallel edge
    for u, v, d in G.edges(data=True):
        a, b = idx[u], idx[v]
        if a == b:
            continue
        length = float(d["length"])
        if mode == "time":
            key, cost = (a, b), length / (_edge_kph(d) / 3.6)
        else:
            key, cost = (min(a, b), max(a, b)), length
        if key not in best or cost < best[key][0]:
            best[key] = (cost, length)
    keys = list(best)
    r = np.array([k[0] for k in keys]); c = np.array([k[1] for k in keys])
    cost = np.array([best[k][0] for k in keys]); length = np.array([best[k][1] for k in keys])
    straight = haversine(la[r], lo[r], la[c], lo[c])
    length = np.maximum(length, straight)        # an edge is never shorter than the straight line
    if mode == "time":
        speed = length / cost                    # m/s per edge, after the length fix
        v_max = float(speed.max())
        extra = {"scale": 1.0 / v_max, "v_max": v_max}
    else:
        cost = length
        r, c = np.r_[r, c], np.r_[c, r]          # store both directions
        cost = np.r_[cost, cost]; length = np.r_[length, length]
        extra = {"scale": 1.0}
    A = sp.csr_matrix((cost, (r, c)), shape=(len(nodes), len(nodes)))
    L = sp.csr_matrix((length, (r, c)), shape=(len(nodes), len(nodes)))
    A.sort_indices(); L.sort_indices()
    out_dir = data_dir(mode); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.npz"
    np.savez_compressed(out, lat=la, lon=lo, indptr=A.indptr, indices=A.indices, weights=A.data,
                        lengths=L.data, directed=mode == "time", **extra)
    return out


def load(name: str, mode: str = "dist") -> RoadGraph:
    f = data_dir(mode) / f"{name}.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run `python scripts/download.py --mode {mode} {name}` first")
    z = np.load(f)
    return RoadGraph(name, z["lat"], z["lon"], z["indptr"], z["indices"], z["weights"],
                     directed=bool(z["directed"]) if "directed" in z else False,
                     scale=float(z["scale"]) if "scale" in z else 1.0,
                     lengths=z["lengths"] if "lengths" in z else None, mode=mode)
