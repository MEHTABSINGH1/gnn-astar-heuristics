"""Summarise results/eval/*.csv into tables, figures and headline numbers.

Method names: dijkstra, haversine, alt_b{k}, const_tau{q}_b{k}, <enc>_tau{q}_s{seed}_b{k}.

Per (city, method) metrics, averaged over queries:
  exp_ratio   geometric mean of expansions / A*-haversine expansions (lower = less search)
  subopt_pct  mean (cost / optimal - 1) x 100
  optimal_pct share of queries whose path is optimal
Learned models: computed per seed, then mean ± std over 3 seeds.

The quantile tau is chosen on the VALIDATION city (patiala) only: per method,
the tau with the fewest expansions among those with mean sub-optimality <= 1%.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

VAL = "patiala"
SEEN = "chandigarh"
UNSEEN = ["amritsar", "dharwad", "mysuru", "bengaluru"]
BUDGETS = [0, 1, 2, 4, 8, 16]
NAME = {"haversine": "A* + haversine", "alt": "A* + ALT (k landmarks)", "const": "haversine × constant",
        "mlp": "MLP (no graph)", "sage": "GraphSAGE", "gat": "GAT", "dijkstra": "Dijkstra"}
STYLE = {"haversine": ("k", "s", "--"), "alt": ("tab:green", "D", "-"), "const": ("tab:purple", "^", "-"),
         "mlp": ("tab:gray", "o", "-"), "sage": ("tab:blue", "o", "-"), "gat": ("tab:orange", "o", "-")}


def parse(m: str) -> dict:
    if m in ("dijkstra", "haversine"):
        return {"base": m, "tau": np.nan, "seed": -1, "budget": 0}
    mm = re.fullmatch(r"(?P<base>[a-z]+)(?:_tau(?P<tau>[0-9.]+))?(?:_s(?P<seed>\d+))?_b(?P<b>\d+)", m)
    if not mm:
        raise ValueError(m)
    return {"base": mm["base"], "tau": float(mm["tau"]) if mm["tau"] else np.nan,
            "seed": int(mm["seed"]) if mm["seed"] else -1, "budget": int(mm["b"])}


def per_query(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.method == "haversine"].set_index("query").expansions
    df = df.copy()
    df["exp_ratio"] = df.expansions.values / np.maximum(base.loc[df["query"]].values, 1)
    df["subopt_pct"] = (df.cost / df.opt - 1) * 100
    df["optimal"] = df.subopt_pct <= 1e-4
    info = pd.DataFrame([parse(m) for m in df.method], index=df.index)
    return pd.concat([df, info], axis=1)


def city_seed_table(q: pd.DataFrame) -> pd.DataFrame:
    return q.groupby(["city", "base", "tau", "seed", "budget"], dropna=False).agg(
        exp_ratio=("exp_ratio", lambda x: float(np.exp(np.log(np.maximum(x, 1e-9)).mean()))),
        expansions=("expansions", "mean"), subopt_pct=("subopt_pct", "mean"),
        subopt_max=("subopt_pct", "max"), optimal_pct=("optimal", lambda x: 100 * x.mean())).reset_index()


def choose_tau(t: pd.DataFrame) -> dict:
    v = t[(t.city == VAL) & t.tau.notna()].groupby(["base", "tau"]).agg(
        exp_ratio=("exp_ratio", "mean"), subopt_pct=("subopt_pct", "mean")).reset_index()
    chosen = {}
    for base, g in v.groupby("base"):
        ok = g[g.subopt_pct <= 1.0]
        pick = (ok if len(ok) else g.nsmallest(1, "subopt_pct")).nsmallest(1, "exp_ratio")
        chosen[base] = float(pick.tau.iloc[0])
    return chosen


def select(t: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    keep = t.tau.isna() | np.array([chosen.get(b) == tau for b, tau in zip(t.base, t.tau)])
    return t[keep]


def over_cities(t: pd.DataFrame, cities) -> pd.DataFrame:
    """Average over cities per seed, then mean ± std over seeds."""
    a = t[t.city.isin(cities)].groupby(["base", "seed", "budget"]).agg(
        exp_ratio=("exp_ratio", "mean"), subopt_pct=("subopt_pct", "mean"),
        optimal_pct=("optimal_pct", "mean")).reset_index()
    return a.groupby(["base", "budget"]).agg(
        exp_ratio=("exp_ratio", "mean"), exp_sd=("exp_ratio", "std"), subopt_pct=("subopt_pct", "mean"),
        subopt_sd=("subopt_pct", "std"), optimal_pct=("optimal_pct", "mean")).reset_index()


def cell(r) -> str:
    sd = f"±{r.exp_sd:.2f}" if pd.notna(r.exp_sd) and r.exp_sd > 0 else ""
    return f"{r.exp_ratio:.2f}{sd} ({r.subopt_pct:.2f}%)"


def row_cells(T: pd.DataFrame, b: str) -> list[str]:
    g = T[T.base == b].set_index("budget")
    if b == "haversine":
        return [cell(g.iloc[0])] + [""] * (len(BUDGETS) - 1)
    return [cell(g.loc[k]) if k in g.index else "–" for k in BUDGETS]


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser()
    p.add_argument("--eval", default="results/eval")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    ev, out = Path(a.eval), Path(a.out)
    figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    q = pd.concat([per_query(pd.read_csv(f)) for f in sorted(ev.glob("*.csv"))])
    t = city_seed_table(q)
    pre = {f.stem.replace("_preprocess", ""): json.loads(f.read_text()) for f in ev.glob("*_preprocess.json")}
    unseen = [c for c in UNSEEN if c in set(t.city)]
    chosen = choose_tau(t) if VAL in set(t.city) else {}
    ts = select(t, chosen)
    U = over_cities(ts, unseen)
    bases = [b for b in ["haversine", "alt", "const", "mlp", "sage", "gat"] if b in set(U.base)]
    head = ["| Method | " + " | ".join(f"k={k}" for k in BUDGETS) + " |", "|---" * (len(BUDGETS) + 1) + "|"]

    md = ["# Results\n",
          f"_{q['query'].nunique()} random queries per city (straight-line distance ≥ 1 km). "
          "Cells: node expansions relative to A* + haversine (geometric mean; lower is better), "
          "with mean path sub-optimality in brackets. Learned models: mean ± std over 3 training seeds._\n",
          "## Road graphs (5 km radius, OpenStreetMap drivable network)\n",
          "| City | Role | Nodes | Edges |", "|---|---|---|---|"]
    role = {SEEN: "train + test (new queries)", VAL: "validation (τ selection)", **{c: "test, unseen" for c in UNSEEN}}
    for c in [SEEN, VAL] + unseen:
        if c in pre:
            md.append(f"| {c.capitalize()} | {role[c]} | {pre[c]['nodes']:,} | {pre[c]['edges']:,} |")
    md += ["\n## Quantile τ chosen on the validation city\n", "| Method | τ |", "|---|---|"]
    md += [f"| {NAME[b]} | {tau} |" for b, tau in chosen.items()]

    md += ["\n## Main result: average over the unseen test cities, by preprocessing budget k\n",
           "k = number of single-source Dijkstra runs allowed in the new city "
           "(ALT: k landmarks; constant: calibrated on k runs, k = 0 uses the training cities; "
           "learned: k = 0 zero-shot, k > 0 fine-tuned on k runs).\n"] + head
    dij = over_cities(t[t.base == "dijkstra"], unseen)
    if len(dij):
        md.append(f"| Dijkstra | {cell(dij.iloc[0])} |" + " |" * (len(BUDGETS) - 1))
    md += [f"| {NAME[b]} | " + " | ".join(row_cells(U, b)) + " |" for b in bases]

    for title, cities in [("Seen city: Chandigarh (new queries)", [SEEN])] + [(c.capitalize(), [c]) for c in unseen]:
        T = over_cities(ts, cities)
        if T.empty:
            continue
        md += [f"\n### {title}\n"] + head + [f"| {NAME[b]} | " + " | ".join(row_cells(T, b)) + " |" for b in bases]

    allt = t[t.tau.notna()].copy()
    allt["base"] = allt.base + "_tau" + allt.tau.astype(str)
    allU = over_cities(allt, unseen)
    md += ["\n## All τ values (unseen-city average, for transparency)\n"] + head
    for b in sorted(allU.base.unique()):
        md.append(f"| {b} | " + " | ".join(row_cells(allU, b)) + " |")

    # Figure: budget curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for b in bases:
        g = U[U.base == b].sort_values("budget")
        col, mk, ls = STYLE[b]
        if b == "haversine":
            for ax, c_ in zip(axes, ["exp_ratio", "subopt_pct"]):
                ax.axhline(g[c_].iloc[0], color=col, ls=ls, lw=1, label=NAME[b])
            continue
        x = [BUDGETS.index(k) for k in g.budget]
        lab = NAME[b] + (f" (τ={chosen[b]})" if b in chosen else "")
        axes[0].errorbar(x, g.exp_ratio, yerr=g.exp_sd.fillna(0), color=col, marker=mk, ls=ls, capsize=2, label=lab)
        axes[1].plot(x, g.subopt_pct, color=col, marker=mk, ls=ls, label=lab)
    for ax in axes:
        ax.set_xticks(range(len(BUDGETS)), [str(k) for k in BUDGETS]); ax.grid(alpha=.3)
        ax.set_xlabel("preprocessing budget k (Dijkstra runs in the new city)")
    axes[0].set_ylabel("expansions / A*-haversine (log scale)"); axes[0].set_yscale("log")
    axes[0].set_title("Search effort (lower is better)", fontsize=10)
    axes[1].set_ylabel("mean path sub-optimality (%)"); axes[1].set_title("Path quality (0 = optimal)", fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Average over {len(unseen)} unseen cities", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "budget_curve.png", dpi=160); plt.close(fig)

    # Figure: speed / optimality trade-off at k = 0 and k = 16, every tau, unseen cities
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, k in zip(axes, [0, 16]):
        for b in ["const", "mlp", "sage", "gat"]:
            pts = allU[allU.base.str.startswith(b + "_tau") & (allU.budget == k)].sort_values("subopt_pct")
            if pts.empty:
                continue
            col, mk, _ = STYLE[b]
            ax.errorbar(pts.subopt_pct, pts.exp_ratio, yerr=pts.exp_sd.fillna(0), color=col, marker=mk,
                        capsize=2, label=NAME[b])
            for _, r in pts.iterrows():
                ax.annotate("τ=" + r.base.split("_tau")[1], (r.subopt_pct, r.exp_ratio), fontsize=6,
                            xytext=(3, 3), textcoords="offset points", color=col)
        a = U[(U.base == "alt") & (U.budget == max(k, 1))]
        if len(a):
            ax.scatter([0], a.exp_ratio, color=STYLE["alt"][0], marker="D", zorder=3,
                       label=f"ALT ({max(k, 1)} landmark{'s' if k > 1 else ''})")
        ax.axhline(1, color="k", ls="--", lw=.8, label=NAME["haversine"])
        ax.set_xlabel("mean path sub-optimality (%)"); ax.grid(alpha=.3)
        ax.set_title(f"k = {k} " + ("(no preprocessing)" if k == 0 else "(16 Dijkstra runs)"), fontsize=10)
    axes[0].set_ylabel("expansions / A*-haversine"); axes[0].legend(fontsize=7)
    fig.suptitle(f"Speed vs. optimality, average over {len(unseen)} unseen cities (down-left is better)", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "tradeoff.png", dpi=160); plt.close(fig)

    # Figure: zero preprocessing per city
    zs = [b for b in ["const", "mlp", "sage", "gat"] if b in bases]
    cities = [SEEN] + unseen
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / max(len(zs), 1)
    for i, b in enumerate(zs):
        vals = []
        for c in cities:
            T = over_cities(ts[ts.base == b], [c])
            vals.append(T[T.budget == 0].exp_ratio.mean())
        ax.bar(np.arange(len(cities)) + (i - (len(zs) - 1) / 2) * w, vals, w, color=STYLE[b][0], label=NAME[b])
    ax.axhline(1, color="k", ls="--", lw=.8, label=NAME["haversine"])
    ax.set_xticks(range(len(cities)), [c.capitalize() + ("*" if c == SEEN else "") for c in cities])
    ax.set_ylabel("expansions / A*-haversine"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=7)
    ax.set_title("No preprocessing in the test city (k = 0; * = seen in training)", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "zero_shot_per_city.png", dpi=160); plt.close(fig)

    def val(b, k, col="exp_ratio"):
        r = U[(U.base == b) & (U.budget == k)]
        return None if r.empty else round(float(r[col].iloc[0]), 4)

    learned = [b for b in ["mlp", "sage", "gat"] if b in bases]
    best0 = min(learned, key=lambda b: val(b, 0)) if learned else None
    def best_learned_upto(k):  # best learned expansion ratio using a budget <= k
        vals = [val(b, j) for b in learned for j in BUDGETS if j <= k and val(b, j) is not None]
        return min(vals) if vals else None
    crossover = next((k for k in BUDGETS[1:] if val("alt", k) is not None and best_learned_upto(k) is not None
                      and val("alt", k) < best_learned_upto(k)), None)
    summary = {"chosen_tau": chosen, "unseen_cities": unseen, "queries_per_city": int(q["query"].nunique()),
               "graphs": {c: {"nodes": pre[c]["nodes"], "edges": pre[c]["edges"]} for c in pre},
               "unseen_avg": {b: {str(k): {"exp_ratio": val(b, k), "subopt_pct": val(b, k, "subopt_pct"),
                                           "optimal_pct": val(b, k, "optimal_pct")}
                                  for k in BUDGETS if val(b, k) is not None} for b in bases + ["dijkstra"]},
               "best_zero_shot_learned": best0,
               "first_budget_where_alt_beats_best_learned": crossover}
    (out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
