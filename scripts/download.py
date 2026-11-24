"""Download and cache city road graphs.

  python scripts/download.py [--mode dist|time] [city ...]     (default: all 5 km cities)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnnastar.graphs import CITIES, download, load  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--mode", default="dist", choices=["dist", "time"])
p.add_argument("cities", nargs="*")
a = p.parse_args()
for name in a.cities or [c for c in CITIES if not c.endswith("_xl")]:
    try:
        download(name, a.mode)
        g = load(name, a.mode)
        extra = f" v_max={1 / g.scale * 3.6:.0f} km/h" if a.mode == "time" else ""
        print(f"{name:13s} nodes={g.n:7d} edges={g.m:7d}{extra}", flush=True)
    except Exception as e:  # keep going; report at the end
        print(f"{name:13s} FAILED: {type(e).__name__}: {e}", flush=True)
