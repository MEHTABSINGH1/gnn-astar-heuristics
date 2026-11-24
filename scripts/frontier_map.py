"""Draw which streets each search method explores for one long query on a real city map.

  python scripts/frontier_map.py --mode time --city bengaluru

Panels: Dijkstra, A* + straight-line bound, ALT (16 landmarks), straight-line x constant,
and the learned GraphSAGE heuristic (zero-shot, tau chosen on the validation city).
Grey = road network, colour = expanded nodes, red = returned route. The query is the
one with the median straight-line distance among the 20 longest random queries, so the
picture is typical rather than cherry-picked; its index is printed and written on the figure.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnnastar.evaluate import queries  # noqa: E402
from gnnastar.graphs import load  # noqa: E402
from gnnastar.model import graph_tensors, heuristic_for_target  # noqa: E402
from gnnastar.search import ALT, astar  # noqa: E402
from gnnastar.train import load_model  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="time", choices=["dist", "time"])
    p.add_argument("--city", default="bengaluru")
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    base = ROOT / "results" / ("time" if a.mode == "time" else "")
    g = load(a.city, a.mode)
    summ = json.loads((base / "summary.json").read_text())
    tau = summ["chosen_tau"].get("sage", 0.2)
    const = json.loads((base / "eval" / f"{a.city}_preprocess.json").read_text())["constants"] \
        if (base / "eval" / f"{a.city}_preprocess.json").exists() else {}
    c0 = const.get(f"const_tau{tau}_b0")

    qs = queries(g, 150, a.seed)
    d = np.array([g.haversine_to(t)[s] for s, t in qs])
    longest = np.argsort(d)[-20:]
    qi = int(longest[np.argsort(d[longest])[len(longest) // 2]])
    s, t = qs[qi]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mp = sorted(glob.glob(str(base / "models" / f"sage_tau{tau}_s0.pt")))[0]
    m = load_model(Path(mp), dev)
    with torch.no_grad():
        z = m.embed(*graph_tensors(g, dev))
    alt = ALT(g, 16, seed=a.seed)
    hb = g.base_heuristic(t)
    unit = "straight-line / v_max" if a.mode == "time" else "haversine"
    methods = [("Dijkstra", None), (f"A* + {unit}", hb), ("A* + ALT (16 landmarks)", alt.heuristic(t))]
    if c0:
        methods.append((f"{unit} × {c0:.2f} (constant)", hb * c0))
    methods.append((f"GraphSAGE heuristic (zero-shot, τ={tau})", heuristic_for_target(m, g, z, t)))
    res = [(name, astar(g, s, t, h, record=True)) for name, h in methods]
    opt = res[0][1].cost

    ei = g.edge_index()
    segs = np.stack([np.c_[g.lon[ei[0]], g.lat[ei[0]]], np.c_[g.lon[ei[1]], g.lat[ei[1]]]], axis=1)
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 5.0))
    for ax, (name, r) in zip(axes, res):
        ax.add_collection(LineCollection(segs, colors="#dcdcdc", linewidths=0.25))
        ex = np.array(r.expanded)
        ax.scatter(g.lon[ex], g.lat[ex], s=3.0, color="#1f77b4", alpha=0.55, linewidths=0)
        pth = np.array(r.path)
        ax.plot(g.lon[pth], g.lat[pth], color="red", lw=1.6)
        ax.scatter(g.lon[[s, t]], g.lat[[s, t]], color="red", s=18, zorder=3)
        sub = (r.cost / opt - 1) * 100
        ax.set_title(f"{name}\n{r.expansions:,} nodes explored · route +{sub:.1f}%", fontsize=9.5)
        ax.set_aspect(1 / np.cos(np.radians(g.lat.mean()))); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(g.lon.min(), g.lon.max()); ax.set_ylim(g.lat.min(), g.lat.max())
    cost = "travel time" if a.mode == "time" else "distance"
    fig.suptitle(f"{a.city.capitalize()} ({g.n:,} nodes, {cost}): nodes explored for the same query "
                 f"(query #{qi}, {d[qi] / 1000:.1f} km straight-line). Blue = explored, red = route", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = base / "figures" / f"frontier_{a.city}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170); plt.close(fig)
    print(out, {n: r.expansions for n, r in res})


if __name__ == "__main__":
    main()
