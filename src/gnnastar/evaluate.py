"""Run every heuristic on the same random queries, across preprocessing budgets.

Usage:
  python -m gnnastar.evaluate --cities patiala amritsar dharwad mysuru bengaluru \
      --models "results/models/*.pt" --queries 150

The question is: given a budget of k single-source Dijkstra runs in a NEW city,
which heuristic makes A* expand the fewest nodes, at what loss of optimality?

Methods (k = budget):
  dijkstra, haversine     no preprocessing, exact
  alt_b{k}                ALT with k landmarks (k >= 1), exact
  const_tau{q}_b{k}       haversine x one detour ratio: the q-quantile measured on k
                          calibration Dijkstras (k = 0: measured on the training cities)
  <model>_b{k}            learned model; k = 0 zero-shot, k > 0 fine-tuned on k
                          calibration Dijkstras
Calibration targets never overlap with query targets.

Writes results/eval/<city>.csv (one row per query x method) and
results/eval/<city>_preprocess.json (one-off preprocessing times, constants).
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graphs import load
from .model import graph_tensors, heuristic_for_target
from .search import ALT, astar, distances_to
from .train import finetune, load_model

ALT_BUDGETS = (1, 2, 4, 8, 16)
CONST_BUDGETS = (0, 1, 2, 4, 8, 16)
LEARNED_BUDGETS = (0, 2, 16)


def queries(g, n: int, seed: int, min_km: float = 1.0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s, t = (int(x) for x in rng.integers(g.n, size=2))
        if s != t and g.haversine_to(t)[s] >= min_km * 1000:
            out.append((s, t))
    return out


def calibration_labels(g, exclude: set[int], seed: int, k: int = 16):
    """k calibration targets; the first j of them form the budget-j label set."""
    rng = np.random.default_rng(seed + 999)
    pool = np.setdiff1d(np.arange(g.n), np.fromiter(exclude, dtype=int))
    ct = rng.choice(pool, size=k, replace=False)
    D = distances_to(g, ct)
    H = np.stack([g.base_heuristic(int(x)) for x in ct])
    return ct, D, H


def evaluate_city(city: str, model_paths: list[str], n_queries: int, seed: int, out_dir: Path,
                  ft_steps: int = 100, device: str | None = None, mode: str = "dist") -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = load(city, mode)
    qs = queries(g, n_queries, seed)
    pre: dict = {"city": city, "nodes": g.n, "edges": g.m, "mode": mode, "directed": g.directed}

    t0 = time.perf_counter(); alt = ALT(g, max(ALT_BUDGETS), seed=seed)
    pre["alt16_s"] = time.perf_counter() - t0
    ct, D, H = calibration_labels(g, {t for _, t in qs}, seed)

    def labels(k):
        return ct[:k], D[:k], H[:k]

    models: dict[str, tuple] = {}
    taus: dict[float, float] = {}
    x, ei, el = graph_tensors(g, device)
    for p in model_paths:
        stem = Path(p).stem
        m = load_model(Path(p), device)
        tau = m.info["tau"]
        taus.setdefault(tau, m.info["train_const_ratio"])
        for k in LEARNED_BUDGETS:
            t0 = time.perf_counter()
            mk = m if k == 0 else finetune(m, g, labels(k), tau, steps=ft_steps, seed=seed)
            with torch.no_grad():
                models[f"{stem}_b{k}"] = (mk, mk.embed(x, ei, el))
            pre[f"{stem}_b{k}_prep_s"] = time.perf_counter() - t0

    consts = {}
    for tau, c0 in taus.items():
        for k in CONST_BUDGETS:
            if k == 0:
                consts[f"const_tau{tau}_b0"] = c0
            else:
                ok = H[:k] > 50 * g.scale
                lr = np.log(np.maximum(D[:k][ok] / H[:k][ok], 1.0))
                consts[f"const_tau{tau}_b{k}"] = float(np.exp(np.quantile(lr, tau)))
    pre["constants"] = consts

    rows = []
    for qi, (s, t) in enumerate(qs):
        t0 = time.perf_counter(); hav = g.base_heuristic(t); hb = time.perf_counter() - t0
        hs = {"dijkstra": (None, 0.0), "haversine": (hav, hb)}
        for k in ALT_BUDGETS:
            t0 = time.perf_counter(); hs[f"alt_b{k}"] = (alt.heuristic(t, k), time.perf_counter() - t0)
        for name, c in consts.items():
            hs[name] = (hav * c, 0.0)
        for name, (m, z) in models.items():
            t0 = time.perf_counter(); h = heuristic_for_target(m, g, z, t)
            hs[name] = (h, time.perf_counter() - t0)
        opt = None
        for name, (h, ht) in hs.items():
            t0 = time.perf_counter(); r = astar(g, s, t, h); st = time.perf_counter() - t0
            if name == "dijkstra":
                opt = r.cost
            rows.append({"city": city, "query": qi, "s": s, "t": t, "method": name,
                         "expansions": r.expansions, "cost": r.cost, "opt": opt,
                         "search_s": st, "heuristic_s": ht})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{city}.csv", index=False)
    (out_dir / f"{city}_preprocess.json").write_text(json.dumps(pre, indent=1))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cities", nargs="+", required=True)
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--queries", type=int, default=150)
    p.add_argument("--ft-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default="results/eval")
    p.add_argument("--mode", default="dist", choices=["dist", "time"])
    a = p.parse_args()
    paths = sorted({f for pat in a.models for f in glob.glob(pat)})
    for c in a.cities:
        if (Path(a.out) / f"{c}.csv").exists():
            print(f"[skip] {c}"); continue
        t0 = time.time()
        df = evaluate_city(c, paths, a.queries, a.seed, Path(a.out), a.ft_steps, mode=a.mode)
        print(f"{c}: {df['query'].nunique()} queries x {df['method'].nunique()} methods "
              f"in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
