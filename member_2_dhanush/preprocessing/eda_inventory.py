"""Step 1: load raw PathSnap trips and build a dataset inventory (EDA).

Usage:
    python data_processing/eda_inventory.py "path/to/Student folder"
Outputs:
    results/eda_inventory.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def load_trip(trip_dir: Path):
    """Return dict with track (n,3), nodes (m,2), arcs (k,2), route (r,)."""
    tid = trip_dir.name
    out = {}
    out["track"] = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)   # lon, lat, t
    out["nodes"] = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)   # lon, lat
    out["arcs"] = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=int, ndmin=2)  # from, to
    route_f = trip_dir / f"{tid}.route"
    out["route"] = (np.loadtxt(route_f, dtype=int, ndmin=1)
                    if route_f.exists() else None)
    return out


def inventory_row(partition: str, trip_dir: Path):
    d = load_trip(trip_dir)
    tr, nodes, arcs, route = d["track"], d["nodes"], d["arcs"], d["route"]
    dt = np.diff(tr[:, 2])
    n = len(nodes)
    adj = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_comp, _ = connected_components(adj, directed=False)
    deg = np.bincount(arcs.ravel(), minlength=n)
    return {
        "partition": partition,
        "trip_id": trip_dir.name,
        "n_points": len(tr),
        "n_nan_track": int(np.isnan(tr).sum()),
        "frac_dt_1s": float((dt == 1).mean()),
        "max_dt": float(dt.max()),
        "n_dt_nonpositive": int((dt <= 0).sum()),
        "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(),
        "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
        "n_nodes": n,
        "n_arcs": len(arcs),
        "n_components": int(n_comp),
        "n_isolated_nodes": int((deg == 0).sum()),
        "arc_idx_ok": bool(arcs.max() < n and arcs.min() >= 0),
        "route_len": None if route is None else len(route),
        "route_idx_ok": None if route is None else bool(route.max() < len(arcs)),
    }


def main(root: str):
    root = Path(root)
    rows = []
    for part in ["train", "student_test"]:
        for trip_dir in sorted((root / part).iterdir()):
            if trip_dir.is_dir():
                rows.append(inventory_row(part, trip_dir))
    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/eda_inventory.csv", index=False)
    print(df.groupby("partition").agg(
        trips=("trip_id", "count"),
        pts_mean=("n_points", "mean"),
        arcs_mean=("n_arcs", "mean"),
        nan=("n_nan_track", "sum"),
        bad_dt=("n_dt_nonpositive", "sum"),
        comps_mean=("n_components", "mean")))
    print("\nWritten: results/eda_inventory.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Student folder")
