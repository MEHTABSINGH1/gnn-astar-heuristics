"""Train every model, evaluate all heuristics, then aggregate.

  python scripts/run_all.py                    # study 1: distance, undirected   -> results/
  python scripts/run_all.py --mode time        # study 2: travel time, one-ways  -> results/time/
  python scripts/run_all.py --quick [--mode …] # tiny smoke run                  -> results/[time/]quick/

Resumable: finished models and evaluated cities are skipped.

Design
  training cities : chandigarh, ludhiana, jalandhar     (Punjab)
  validation city : patiala                             (chooses the quantile tau; never a test city)
  test cities     : chandigarh (seen city, new queries), amritsar (unseen, Punjab),
                    dharwad, mysuru, bengaluru           (unseen, Karnataka: a different region)
  scaling (time)  : bengaluru_xl (12 km radius), ludhiana_xl (10 km) with the selected tau only
  models          : {MLP (no message passing), GraphSAGE, GAT} x tau {0.05, 0.2} x 3 seeds
  budgets         : see gnnastar/evaluate.py (0-16 Dijkstra runs in each test city)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnnastar.graphs import data_dir  # noqa: E402
from gnnastar.train import train  # noqa: E402

TRAIN = ["chandigarh", "ludhiana", "jalandhar"]
VAL = "patiala"
TEST = ["patiala", "chandigarh", "amritsar", "dharwad", "mysuru", "bengaluru"]  # patiala = validation
XL = ["ludhiana_xl", "bengaluru_xl"]
GRID = [(e, t) for e in ("mlp", "sage", "gat") for t in (0.05, 0.2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mode", default="dist", choices=["dist", "time"])
    p.add_argument("--xl-queries", type=int, default=60)
    a = p.parse_args()
    seeds, steps, queries = ([0], 100, 8) if a.quick else ([0, 1, 2], 1500, 150)
    grid = [("mlp", 0.05), ("sage", 0.2)] if a.quick else GRID
    base = ROOT / "results" / ("time" if a.mode == "time" else "")
    base = base / "quick" if a.quick else base
    mdir, edir = base / "models", base / "eval"
    for enc, tau in grid:
        for s in seeds:
            out = mdir / f"{enc}_tau{tau}_s{s}.pt"
            if out.exists():
                print(f"[skip] {out.name}"); continue
            r = train(enc, TRAIN, VAL, tau, s, steps=steps, out=out, mode=a.mode)
            i = r["info"]
            print(f"[trained] {out.name}  val pinball={i['best_val_pinball']:.4f}  "
                  f"{i['train_seconds']:.0f}s", flush=True)
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    ev = [py, "-m", "gnnastar.evaluate", "--mode", a.mode, "--out", str(edir),
          "--ft-steps", "10" if a.quick else "100"]
    subprocess.run(ev + ["--cities", *(TEST[:3] if a.quick else TEST), "--models", str(mdir / "*.pt"),
                         "--queries", str(queries)], check=True, cwd=ROOT, env=env)
    agg = [py, str(ROOT / "scripts/aggregate.py"), "--eval", str(edir), "--out", str(base)]
    subprocess.run(agg, check=True, cwd=ROOT)
    xl = [c for c in XL if (data_dir(a.mode) / f"{c}.npz").exists()]
    if a.mode == "time" and xl and not a.quick:
        # large graphs: only the tau each method selected on the validation city, fewer queries
        import json
        chosen = json.loads((base / "summary.json").read_text())["chosen_tau"]
        pats = [str(mdir / f"{enc}_tau{tau}_s*.pt") for enc, tau in chosen.items() if enc != "const"]
        subprocess.run(ev + ["--cities", *xl, "--models", *pats, "--queries", str(a.xl_queries)],
                       check=True, cwd=ROOT, env=env)
        subprocess.run(agg, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
