"""Train every model, evaluate all heuristics, then aggregate.

  python scripts/run_all.py            # full experiment (resumable: finished models are skipped)
  python scripts/run_all.py --quick    # tiny smoke run

Design
  training cities : chandigarh, ludhiana, jalandhar     (Punjab)
  validation city : patiala                             (chooses the quantile tau; never a test city)
  test cities     : chandigarh (seen city, new queries), amritsar (unseen, Punjab),
                    dharwad, mysuru, bengaluru           (unseen, Karnataka: a different region)
  models          : {MLP (no message passing), GraphSAGE, GAT} x tau {0.05, 0.2} x 3 seeds
  budgets         : see gnnastar/evaluate.py (0-16 Dijkstra runs in each test city)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnnastar.train import train  # noqa: E402

TRAIN = ["chandigarh", "ludhiana", "jalandhar"]
VAL = "patiala"
TEST = ["patiala", "chandigarh", "amritsar", "dharwad", "mysuru", "bengaluru"]  # patiala = validation
GRID = [(e, t) for e in ("mlp", "sage", "gat") for t in (0.05, 0.2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    seeds, steps, queries = ([0], 100, 8) if a.quick else ([0, 1, 2], 1500, 150)
    grid = [("mlp", 0.05), ("sage", 0.2)] if a.quick else GRID
    mdir = ROOT / ("results/quick/models" if a.quick else "results/models")
    edir = ROOT / ("results/quick/eval" if a.quick else "results/eval")
    for enc, tau in grid:
        for s in seeds:
            out = mdir / f"{enc}_tau{tau}_s{s}.pt"
            if out.exists():
                print(f"[skip] {out.name}"); continue
            r = train(enc, TRAIN, VAL, tau, s, steps=steps, out=out)
            i = r["info"]
            print(f"[trained] {out.name}  val pinball={i['best_val_pinball']:.4f}  "
                  f"{i['train_seconds']:.0f}s", flush=True)
    py = sys.executable
    subprocess.run([py, "-m", "gnnastar.evaluate", "--cities", *(TEST[:3] if a.quick else TEST),
                    "--models", str(mdir / "*.pt"), "--queries", str(queries), "--out", str(edir),
                    "--ft-steps", "10" if a.quick else "100"],
                   check=True, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
    subprocess.run([py, str(ROOT / "scripts/aggregate.py"), "--eval", str(edir),
                    "--out", str(edir.parent)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
