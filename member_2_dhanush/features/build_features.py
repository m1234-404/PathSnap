"""Feature engineering + common processed dataset for model training.

One row = one (GPS observation, candidate arc) pair.
Target `label` = 1 if the candidate arc is the reference arc for that observation.

Run from the repo root:
    python -m features.build_features "path/to/Student folder"
Outputs (data/processed/):
    train.parquet, student_test.parquet     candidate-level feature tables
    train_obs.parquet, student_test_obs.parquet   observation-level tables (labels, coords)
    candidate_stats_<partition>.csv         per-trip miss rate / alignment quality
    feature_cols.json                       the exact model input columns

The same functions are reused by the API for the evaluator trips, with
with_labels=False so that no ground truth enters the prediction pipeline.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.candidates import (align_route_to_points, arc_endpoints,
                                      generate_candidates)
from preprocessing.loader import LocalProjector, load_trip

RADIUS_M = 100.0   # candidate search radius (chosen from miss-rate sweep)
TOP_K = 20         # max candidates per GPS point
MAX_MISS = 0.30    # trips whose reference route cannot be matched to the GPS track
                   # (candidate miss rate above this) are flagged trip_ok=False

FEATURE_COLS = [
    "dist", "dist_rel", "dist_rank", "n_cands",       # geometry vs. other candidates
    "frac", "cross", "arc_len",                       # position on / shape of the arc
    "cos_heading", "heading_valid", "move_along",     # direction agreement
    "speed", "log_dt_prev", "log_dt_next",            # kinematics / sampling gaps
    "in_deg_from", "out_deg_to",                      # junction complexity
    "same_prev", "cont_prev", "pred_prev", "rev_prev",  # topology vs. previous best arc
    "same_next", "cont_next", "pred_next", "rev_next",  # topology vs. next best arc
]


def _gps_kinematics(P, t):
    """Heading (unit vector), validity flag, speed, dt_prev, dt_next per GPS point."""
    n = len(P)
    idx = np.arange(n)
    lo, hi = np.maximum(idx - 2, 0), np.minimum(idx + 2, n - 1)
    h = P[hi] - P[lo]                         # smoothed direction of travel
    hn = np.linalg.norm(h, axis=1)
    valid = hn > 2.0                          # <2 m over the window = stationary/noise
    hu = np.where(valid[:, None], h / np.maximum(hn, 1e-9)[:, None], 0.0)

    prev = np.maximum(idx - 1, 0)
    nxt = np.minimum(idx + 1, n - 1)
    dt_prev = t - t[prev]
    dt_next = t[nxt] - t
    dt_prev[0] = dt_next[0]
    dt_next[-1] = dt_prev[-1]
    dt_prev = np.maximum(dt_prev, 1.0)
    dt_next = np.maximum(dt_next, 1.0)
    step = np.linalg.norm(P - P[prev], axis=1)
    speed = np.clip(step / dt_prev, 0, 70.0)
    return hu, valid, speed, dt_prev, dt_next


def _relations(cand_arc, ref_arc, f, to):
    """Topology of candidate arcs vs a reference arc (nearest arc of neighbour point)."""
    ok = ref_arc >= 0
    b = np.where(ok, ref_arc, 0)
    same = ok & (cand_arc == b)
    rev = ok & (f[cand_arc] == to[b]) & (to[cand_arc] == f[b]) & ~same
    cont = ok & (to[b] == f[cand_arc]) & ~same & ~rev      # ref arc then candidate
    pred = ok & (to[cand_arc] == f[b]) & ~same & ~rev      # candidate then ref arc
    return same, cont, pred, rev


def build_trip_table(d, with_labels=True, radius=RADIUS_M, k=TOP_K):
    """Candidate-level feature table (+ observation-level table) for one trip dict."""
    track, nodes, arcs = d["track"], d["nodes"], d["arcs"]
    pj = LocalProjector.from_track(track)
    P = pj.forward(track[:, 0], track[:, 1])
    t = track[:, 2]
    N = pj.forward(nodes[:, 0], nodes[:, 1])
    A, B = arc_endpoints(N, arcs)
    n = len(P)

    cand = generate_candidates(P, A, B, radius=radius, k=k)
    obs = cand["obs_idx"].to_numpy()
    arc = cand["arc_id"].to_numpy()

    hu, hvalid, speed, dt_prev, dt_next = _gps_kinematics(P, t)
    AB = B[arc] - A[arc]
    arc_len = np.linalg.norm(AB, axis=1)
    adir = AB / np.maximum(arc_len, 1e-9)[:, None]

    cand["cos_heading"] = (adir * hu[obs]).sum(1)
    cand["heading_valid"] = hvalid[obs].astype(np.float32)
    prev = np.maximum(obs - 1, 0)
    cand["move_along"] = np.clip(((P[obs] - P[prev]) * adir).sum(1), -100, 100)
    cand["arc_len"] = arc_len
    cand["speed"] = speed[obs]
    cand["log_dt_prev"] = np.log1p(dt_prev[obs])
    cand["log_dt_next"] = np.log1p(dt_next[obs])

    g = cand.groupby("obs_idx")["dist"]
    cand["dist_rel"] = cand["dist"] - g.transform("min")
    cand["n_cands"] = g.transform("size")
    cand["dist_rank"] = cand["rank"]

    f, to = arcs[:, 0], arcs[:, 1]
    in_deg = np.bincount(to, minlength=len(nodes))     # arcs entering a node
    out_deg = np.bincount(f, minlength=len(nodes))     # arcs leaving a node
    cand["in_deg_from"] = in_deg[f[arc]]
    cand["out_deg_to"] = out_deg[to[arc]]

    best = np.full(n, -1, dtype=np.int64)               # nearest arc of every point
    top = cand[cand["rank"] == 0]
    best[top["obs_idx"].to_numpy()] = top["arc_id"].to_numpy()
    for name, ref in (("prev", np.where(obs > 0, best[np.maximum(obs - 1, 0)], -1)),
                      ("next", np.where(obs < n - 1, best[np.minimum(obs + 1, n - 1)], -1))):
        s, c, p, r = _relations(arc, ref, f, to)
        cand[f"same_{name}"], cand[f"cont_{name}"] = s, c
        cand[f"pred_{name}"], cand[f"rev_{name}"] = p, r

    obs_df = pd.DataFrame({"trip_id": d["trip_id"], "obs_idx": np.arange(n),
                           "lon": track[:, 0], "lat": track[:, 1], "t": t})
    stats = {"trip_id": d["trip_id"], "n_obs": n, "n_rows": len(cand)}
    if with_labels and d["route"] is not None:
        true_arc, _, adist = align_route_to_points(P, A, B, d["route"])
        cand["label"] = (arc == true_arc[obs]).astype(np.int8)
        obs_df["true_arc"] = true_arc
        obs_df["in_cands"] = obs_df["obs_idx"].isin(cand.loc[cand["label"] == 1, "obs_idx"])
        stats.update(candidate_miss_rate=float(1 - obs_df["in_cands"].mean()),
                     align_dist_median=float(np.median(adist)),
                     align_dist_p95=float(np.percentile(adist, 95)))

    cand.insert(0, "trip_id", d["trip_id"])
    keep = ["trip_id", "obs_idx", "arc_id"] + FEATURE_COLS
    if "label" in cand:
        keep.append("label")
    cand = cand[keep]
    cand[FEATURE_COLS] = cand[FEATURE_COLS].astype(np.float32)
    return cand, obs_df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help='path to "Student folder" (contains train/, student_test/)')
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--limit", type=int, default=None, help="only first N trips (debug)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for part in ["train", "student_test"]:
        dirs = sorted(p for p in (Path(a.root) / part).iterdir() if p.is_dir())[:a.limit]
        tabs, obss, stats = [], [], []
        t0 = time.time()
        for i, td in enumerate(dirs, 1):
            c, o, s = build_trip_table(load_trip(td))
            tabs.append(c); obss.append(o); stats.append(s)
            print(f"[{part}] {i}/{len(dirs)} trip {td.name}: {len(o)} obs, {len(c)} rows, "
                  f"miss={s['candidate_miss_rate']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        st = pd.DataFrame(stats)
        st["trip_ok"] = st["candidate_miss_rate"] <= MAX_MISS
        ok = dict(zip(st["trip_id"], st["trip_ok"]))
        tab, obs = pd.concat(tabs, ignore_index=True), pd.concat(obss, ignore_index=True)
        tab["trip_ok"] = tab["trip_id"].map(ok)      # NOT a model feature: filter on it
        obs["trip_ok"] = obs["trip_id"].map(ok)
        tab.to_parquet(out / f"{part}.parquet", index=False)
        obs.to_parquet(out / f"{part}_obs.parquet", index=False)
        st.to_csv(out / f"candidate_stats_{part}.csv", index=False)
        print("flagged trip_ok=False:", st.loc[~st["trip_ok"], "trip_id"].tolist())
        print(f"\n== {part}: mean candidate miss rate {st['candidate_miss_rate'].mean():.4f}, "
              f"worst {st['candidate_miss_rate'].max():.4f}\n")
    (out / "feature_cols.json").write_text(json.dumps(FEATURE_COLS, indent=2))


if __name__ == "__main__":
    main()
