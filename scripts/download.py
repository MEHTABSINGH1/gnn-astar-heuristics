"""Download and cache city road graphs:  python scripts/download.py [city ...]  (default: all)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnnastar.graphs import CITIES, download, load  # noqa: E402

for name in sys.argv[1:] or list(CITIES):
    try:
        download(name)
        g = load(name)
        print(f"{name:11s} nodes={g.n:6d} edges={g.m:6d}")
    except Exception as e:  # keep going; report at the end
        print(f"{name:11s} FAILED: {type(e).__name__}: {e}")
