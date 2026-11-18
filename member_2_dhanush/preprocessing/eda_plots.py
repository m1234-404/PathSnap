"""EDA figures for the report (all generated from the real data / processed tables).

Run from the repo root, after eda_inventory.py and features.build_features:
    python preprocessing/eda_inventory.py "<root>"
    python -m features.build_features "<root>"
    python -m preprocessing.eda_plots "<root>"
Figures are saved to results/figures/. Each figure names the partition it uses.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing.candidates import arc_endpoints
from preprocessing.loader import LocalProjector, load_trip

OUT = Path("results/figures")
PROC = Path("data/processed")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print("saved", OUT / name)


def fig_inventory(inv):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        s = inv[inv.partition == part]
        ax[0].hist(s.n_points, bins=25, alpha=.6, color=c, label=f"{part} ({len(s)} trips)")
        ax[1].hist(s.frac_dt_1s, bins=20, alpha=.6, color=c, label=part)
        ax[2].scatter(s.n_nodes, s.n_components, alpha=.7, color=c, label=part)
    ax[0].set(title="GPS points per trip", xlabel="points", ylabel="trips")
    ax[1].set(title="Share of exactly-1 s time steps", xlabel="fraction of steps")
    ax[2].set(title="Road network: nodes vs connected components", xlabel="nodes",
              ylabel="components", xscale="log")
    for a in ax:
        a.legend()
    save(fig, "eda_inventory_overview.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(inv.max_dt.clip(lower=1)), bins=25, color="tab:red", alpha=.8)
    ax.set(title="Largest time gap per trip (train + student_test)",
           xlabel="log10(max gap in seconds)", ylabel="trips")
    save(fig, "eda_max_time_gap.png")


def fig_candidate_quality():
    st = pd.read_csv(PROC / "candidate_stats_train.csv").sort_values("candidate_miss_rate")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(st)), st.candidate_miss_rate,
           color=np.where(st.trip_ok, "tab:blue", "tab:red"))
    ax.axhline(0.30, ls="--", color="k", lw=1)
    ax.set(title="Candidate miss rate per train trip (red = flagged trip_ok=False)",
           xlabel="train trips (sorted)", ylabel="miss rate (true arc not in candidates)")
    save(fig, "eda_candidate_miss_rate_train.png")


def fig_trajectories(root):
    cases = [("train", "00000003", "typical"), ("train", "00000001", "typical"),
             ("train", "00000054", "ANOMALY: GPS off the network"),
             ("train", "00000033", "ANOMALY: route far from GPS")]
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for a, (part, tid, note) in zip(ax.ravel(), cases):
        d = load_trip(Path(root) / part / tid)
        pj = LocalProjector.from_track(d["track"])
        P = pj.forward(d["track"][:, 0], d["track"][:, 1])
        N = pj.forward(d["nodes"][:, 0], d["nodes"][:, 1])
        A, B = arc_endpoints(N, d["arcs"])
        lo, hi = P.min(0) - 150, P.max(0) + 150
        inb = ((A[:, 0] > lo[0]) & (A[:, 0] < hi[0]) & (A[:, 1] > lo[1]) & (A[:, 1] < hi[1]))
        for i in np.where(inb)[0][:15000]:
            a.plot([A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]], color="#bbbbbb", lw=.5)
        r = d["route"]
        for i in r:
            a.plot([A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]], color="green", lw=2)
        a.scatter(P[:, 0], P[:, 1], s=2, color="blue", zorder=3)
        a.set(title=f"{part} {tid}: {note}", aspect="equal", xlabel="east (m)", ylabel="north (m)")
    save(fig, "eda_trajectories_route_network.png")


def fig_features():
    df = pd.read_parquet(PROC / "train.parquet")
    df = df[df.trip_ok]
    cols = ["dist", "dist_rel", "cos_heading", "move_along", "speed", "arc_len"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for a, c in zip(ax.ravel(), cols):
        lo, hi = df[c].quantile([.005, .995])
        bins = np.linspace(lo, hi, 50)
        a.hist(df.loc[df.label == 0, c], bins=bins, alpha=.55, density=True, label="label 0 (not true arc)")
        a.hist(df.loc[df.label == 1, c], bins=bins, alpha=.55, density=True, label="label 1 (true arc)")
        a.set_title(c)
    ax[0, 0].legend()
    fig.suptitle(f"Feature distributions by label (train, trip_ok only). "
                 f"Positive rate = {df.label.mean():.3f}")
    save(fig, "eda_feature_distributions.png")


if __name__ == "__main__":
    root = sys.argv[1]
    inv = pd.read_csv("results/eda_inventory.csv")
    inv["partition"] = inv["partition"].astype(str)
    fig_inventory(inv)
    fig_candidate_quality()
    fig_trajectories(root)
    fig_features()
