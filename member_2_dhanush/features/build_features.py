
"""Feature engineering + common processed dataset for model training.

One row = one (GPS observation, candidate arc) pair.
Target `label` = 1 if the candidate arc is the reference arc for that observation.

Run from the repo root:
    python -m features.build_features "path/to/Student folder"

Outputs (data/processed/):
    train.parquet, student_test.parquet
    train_obs.parquet, student_test_obs.parquet
    candidate_stats_<partition>.csv
    feature_cols.json

The same functions are reused by the API for evaluator trips,
with with_labels=False so that no ground truth enters prediction.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.candidates import (
    align_route_to_points,
    arc_endpoints,
    generate_candidates,
)
from preprocessing.loader import LocalProjector, load_trip


RADIUS_M = 100.0
TOP_K = 20
MAX_MISS = 0.30

# Maximum allowed interval for calculating movement and heading.
# Time-gap features themselves are preserved.
MAX_GAP_SECONDS = 60.0

FEATURE_COLS = [
    "dist", "dist_rel", "dist_rank", "n_cands",
    "frac", "cross", "arc_len",
    "cos_heading", "heading_valid", "move_along",
    "speed", "log_dt_prev", "log_dt_next",
    "in_deg_from", "out_deg_to",
    "same_prev", "cont_prev", "pred_prev", "rev_prev",
    "same_next", "cont_next", "pred_next", "rev_next",
]


def _gps_kinematics(P, t):
    """Calculate heading, speed and time gaps without crossing large gaps."""

    n = len(P)
    idx = np.arange(n)

    if n == 0:
        empty = np.empty(0, dtype=float)
        return empty.reshape(0, 2), np.zeros(0, dtype=bool), empty, empty, empty

    # Time differences between consecutive GPS observations
    gaps = np.diff(t)

    # Previous and next observation indices
    prev = np.maximum(idx - 1, 0)
    nxt = np.minimum(idx + 1, n - 1)

    dt_prev = t - t[prev]
    dt_next = t[nxt] - t

    if n > 1:
        dt_prev[0] = dt_next[0]
        dt_next[-1] = dt_prev[-1]

    dt_prev = np.maximum(dt_prev, 1.0)
    dt_next = np.maximum(dt_next, 1.0)

    # Identify valid adjacent observations
    prev_valid = np.zeros(n, dtype=bool)
    prev_valid[1:] = gaps <= MAX_GAP_SECONDS

    # Calculate speed only across valid previous intervals
    step = np.linalg.norm(P - P[prev], axis=1)

    speed = np.zeros(n, dtype=float)
    speed[prev_valid] = np.clip(
        step[prev_valid] / dt_prev[prev_valid],
        0,
        70.0,
    )

    # Calculate heading using observations up to two positions away
    lo = np.maximum(idx - 2, 0)
    hi = np.minimum(idx + 2, n - 1)

    h = P[hi] - P[lo]
    hn = np.linalg.norm(h, axis=1)

    # A heading window must not cross a large time gap
    window_valid = np.ones(n, dtype=bool)

    for i in range(n):
        if hi[i] > lo[i]:
            window_valid[i] = np.all(
                gaps[lo[i]:hi[i]] <= MAX_GAP_SECONDS
            )

    valid = (hn > 2.0) & window_valid

    hu = np.zeros_like(h, dtype=float)
    hu[valid] = h[valid] / hn[valid, None]

    return hu, valid, speed, dt_prev, dt_next


def _relations(cand_arc, ref_arc, f, to):
    """Topology of candidate arcs vs a reference arc."""

    ok = ref_arc >= 0
    b = np.where(ok, ref_arc, 0)

    same = ok & (cand_arc == b)

    rev = (
        ok
        & (f[cand_arc] == to[b])
        & (to[cand_arc] == f[b])
        & ~same
    )

    cont = (
        ok
        & (to[b] == f[cand_arc])
        & ~same
        & ~rev
    )

    pred = (
        ok
        & (to[cand_arc] == f[b])
        & ~same
        & ~rev
    )

    return same, cont, pred, rev


def build_trip_table(d, with_labels=True, radius=RADIUS_M, k=TOP_K):
    """Build candidate-level and observation-level tables for one trip."""

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

    # Candidate arc geometry
    AB = B[arc] - A[arc]
    arc_len = np.linalg.norm(AB, axis=1)
    adir = AB / np.maximum(arc_len, 1e-9)[:, None]

    # Heading agreement
    cand["cos_heading"] = (adir * hu[obs]).sum(1)
    cand["heading_valid"] = hvalid[obs].astype(np.float32)

    # Movement along candidate arc.
    # Do not use a previous observation across a large time gap.
    prev = np.maximum(obs - 1, 0)

    move_valid = (obs > 0) & (dt_prev[obs] <= MAX_GAP_SECONDS)

    move = np.zeros(len(obs), dtype=float)

    move[move_valid] = (
        (
            P[obs[move_valid]]
            - P[prev[move_valid]]
        )
        * adir[move_valid]
    ).sum(1)

    cand["move_along"] = np.clip(move, -100, 100)

    cand["arc_len"] = arc_len
    cand["speed"] = speed[obs]

    # Preserve time-gap features, including large gaps
    cand["log_dt_prev"] = np.log1p(dt_prev[obs])
    cand["log_dt_next"] = np.log1p(dt_next[obs])

    # Candidate distance features
    g = cand.groupby("obs_idx")["dist"]

    cand["dist_rel"] = cand["dist"] - g.transform("min")
    cand["n_cands"] = g.transform("size")
    cand["dist_rank"] = cand["rank"]

    # Road-network degree features
    f, to = arcs[:, 0], arcs[:, 1]

    in_deg = np.bincount(to, minlength=len(nodes))
    out_deg = np.bincount(f, minlength=len(nodes))

    cand["in_deg_from"] = in_deg[f[arc]]
    cand["out_deg_to"] = out_deg[to[arc]]

    # Best candidate arc for every GPS observation
    best = np.full(n, -1, dtype=np.int64)

    top = cand[cand["rank"] == 0]

    best[top["obs_idx"].to_numpy()] = top["arc_id"].to_numpy()

    # Previous and next road-network relationships
    for name, ref in (
        (
            "prev",
            np.where(
                obs > 0,
                best[np.maximum(obs - 1, 0)],
                -1,
            ),
        ),
        (
            "next",
            np.where(
                obs < n - 1,
                best[np.minimum(obs + 1, n - 1)],
                -1,
            ),
        ),
    ):
        s, c, p, r = _relations(arc, ref, f, to)

        cand[f"same_{name}"] = s
        cand[f"cont_{name}"] = c
        cand[f"pred_{name}"] = p
        cand[f"rev_{name}"] = r

    # Observation-level table
    obs_df = pd.DataFrame({
        "trip_id": d["trip_id"],
        "obs_idx": np.arange(n),
        "lon": track[:, 0],
        "lat": track[:, 1],
        "t": t,
    })

    stats = {
        "trip_id": d["trip_id"],
        "n_obs": n,
        "n_rows": len(cand),
    }

    # Ground-truth labels are only generated for labelled training data
    if with_labels and d["route"] is not None:

        true_arc, _, adist = align_route_to_points(
            P, A, B, d["route"]
        )

        cand["label"] = (
            arc == true_arc[obs]
        ).astype(np.int8)

        obs_df["true_arc"] = true_arc

        obs_df["in_cands"] = obs_df["obs_idx"].isin(
            cand.loc[
                cand["label"] == 1,
                "obs_idx",
            ]
        )

        stats.update(
            candidate_miss_rate=float(
                1 - obs_df["in_cands"].mean()
            ),
            align_dist_median=float(np.median(adist)),
            align_dist_p95=float(np.percentile(adist, 95)),
        )

    # Final feature table
    cand.insert(0, "trip_id", d["trip_id"])

    keep = ["trip_id", "obs_idx", "arc_id"] + FEATURE_COLS

    if "label" in cand:
        keep.append("label")

    cand = cand[keep]

    cand[FEATURE_COLS] = cand[FEATURE_COLS].astype(np.float32)

    return cand, obs_df, stats


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "root",
        help='path to "Student folder" (contains train/, student_test/)',
    )

    ap.add_argument("--out", default="data/processed")

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only first N trips (debug)",
    )

    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for part in ["train", "student_test"]:

        dirs = sorted(
            p for p in (Path(a.root) / part).iterdir()
            if p.is_dir()
        )[:a.limit]

        tabs, obss, stats = [], [], []

        t0 = time.time()

        for i, td in enumerate(dirs, 1):

            c, o, s = build_trip_table(load_trip(td))

            tabs.append(c)
            obss.append(o)
            stats.append(s)

            print(
                f"[{part}] {i}/{len(dirs)} trip {td.name}: "
                f"{len(o)} obs, {len(c)} rows, "
                f"miss={s['candidate_miss_rate']:.3f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        st = pd.DataFrame(stats)

        st["trip_ok"] = (
            st["candidate_miss_rate"] <= MAX_MISS
        )

        ok = dict(zip(st["trip_id"], st["trip_ok"]))

        tab = pd.concat(tabs, ignore_index=True)
        obs = pd.concat(obss, ignore_index=True)

        tab["trip_ok"] = tab["trip_id"].map(ok)
        obs["trip_ok"] = obs["trip_id"].map(ok)

        tab.to_parquet(
            out / f"{part}.parquet",
            index=False,
        )

        obs.to_parquet(
            out / f"{part}_obs.parquet",
            index=False,
        )

        st.to_csv(
            out / f"candidate_stats_{part}.csv",
            index=False,
        )

        print(
            "flagged trip_ok=False:",
            st.loc[~st["trip_ok"], "trip_id"].tolist(),
        )

        print(
            f"\n== {part}: mean candidate miss rate "
            f"{st['candidate_miss_rate'].mean():.4f}, "
            f"worst {st['candidate_miss_rate'].max():.4f}\n"
        )

    (out / "feature_cols.json").write_text(
        json.dumps(FEATURE_COLS, indent=2)
    )


if __name__ == "__main__":
    main()
