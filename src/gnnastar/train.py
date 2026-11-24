"""Train a heuristic network on several cities and save it.

Labels: for K random targets per city we run one exact Dijkstra from the
target, which gives the true road distance from *every* node to it. Pairs
(v, t) are then sampled from these rows, so labels are cheap and exact.
Validation targets are disjoint from training targets.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .graphs import load
from .model import HeuristicNet, graph_tensors, pair_features, pinball
from .search import distances_to


def make_labels(g, n_targets: int, rng):
    t = rng.choice(g.n, size=n_targets, replace=False)
    D = distances_to(g, t)                       # (n_targets, n)
    H = np.stack([g.haversine_to(int(x)) for x in t])
    return t, D, H


def sample_pairs(g, t, D, H, n, rng):
    ti = rng.integers(len(t), size=n)
    v = rng.integers(g.n, size=n)
    keep = H[ti, v] > 50.0                       # skip near-identical pairs (ratio ill-defined)
    ti, v = ti[keep], v[keep]
    y = np.log(np.maximum(D[ti, v] / H[ti, v], 1.0))
    return v, t[ti], y.astype(np.float32)


def train(encoder: str, cities: list[str], val_city: str, tau: float, seed: int,
          steps: int = 1500, batch: int = 4096, lr: float = 2e-3, out: Path | None = None,
          device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    graphs = [load(c) for c in cities]
    tensors = [graph_tensors(g, device) for g in graphs]
    labels = [make_labels(g, 150, rng) for g in graphs]
    gv = load(val_city)
    tv = graph_tensors(gv, device)
    lv = make_labels(gv, 40, np.random.default_rng(10_000 + seed))
    vv, vt, vy = sample_pairs(gv, *lv, 20_000, np.random.default_rng(1))
    vpf = torch.tensor(pair_features(gv, vv, vt), device=device)

    model = HeuristicNet(encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    best, best_state, hist = np.inf, None, []
    t0 = time.time()
    for step in range(1, steps + 1):
        k = step % len(graphs)
        g, (x, ei, el) = graphs[k], tensors[k]
        v, t, y = sample_pairs(g, *labels[k], batch, rng)
        pf = torch.tensor(pair_features(g, v, t), device=device)
        model.train()
        z = model.embed(x, ei, el)
        pred = model.log_ratio(z, torch.as_tensor(v, device=device), torch.as_tensor(t, device=device), pf)
        loss = pinball(pred, torch.as_tensor(y, device=device), tau)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                zv = model.embed(*tv)
                pv = model.log_ratio(zv, torch.as_tensor(vv, device=device),
                                     torch.as_tensor(vt, device=device), vpf)
                vl = pinball(pv, torch.as_tensor(vy, device=device), tau).item()
                over = (pv.cpu().numpy() > vy).mean()   # share of over-estimates
            hist.append({"step": step, "train": loss.item(), "val": vl, "val_overestimate": float(over)})
            if vl < best:
                best, best_state = vl, {k_: w.detach().cpu().clone() for k_, w in model.state_dict().items()}
    model.load_state_dict(best_state)
    # zero-budget constant baseline: the tau-quantile detour ratio pooled over training cities
    all_logr = np.concatenate([np.log(np.maximum(D[H > 50] / H[H > 50], 1.0)) for _, D, H in labels])
    info = {"encoder": encoder, "cities": cities, "val_city": val_city, "tau": tau, "seed": seed,
            "train_const_ratio": float(np.exp(np.quantile(all_logr, tau))),
            "steps": steps, "best_val_pinball": best, "train_seconds": time.time() - t0,
            "params": sum(p.numel() for p in model.parameters()), "history": hist}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "info": info}, out)
        out.with_suffix(".json").write_text(json.dumps(info, indent=1))
    return {"model": model, "info": info}


def finetune(model: HeuristicNet, g, labels, tau: float, steps: int = 200, lr: float = 5e-4,
             batch: int = 4096, seed: int = 0) -> HeuristicNet:
    """Adapt a trained model to a new city using a few exact Dijkstra rows (``labels``)."""
    import copy
    device = next(model.parameters()).device
    m = copy.deepcopy(model).train()
    rng = np.random.default_rng(seed)
    x, ei, el = graph_tensors(g, device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(steps):
        v, t, y = sample_pairs(g, *labels, batch, rng)
        pf = torch.tensor(pair_features(g, v, t), device=device)
        pred = m.log_ratio(m.embed(x, ei, el), torch.as_tensor(v, device=device),
                           torch.as_tensor(t, device=device), pf)
        loss = pinball(pred, torch.as_tensor(y, device=device), tau)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m.eval()


def load_model(path: Path, device: str = "cpu") -> HeuristicNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = HeuristicNet(ck["info"]["encoder"]).to(device)
    m.load_state_dict(ck["state"])
    m.info = ck["info"]
    return m.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="sage", choices=["mlp", "sage", "gat"])
    p.add_argument("--cities", nargs="+", default=["chandigarh", "ludhiana", "jalandhar"])
    p.add_argument("--val-city", default="patiala")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = Path(a.out or f"results/models/{a.encoder}_tau{a.tau}_s{a.seed}.pt")
    r = train(a.encoder, a.cities, a.val_city, a.tau, a.seed, a.steps, out=out)
    print(json.dumps({k: v for k, v in r["info"].items() if k != "history"}, indent=1))


if __name__ == "__main__":
    main()
