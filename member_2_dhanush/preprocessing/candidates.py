"""Candidate road-segment generation and reference-label alignment.

Problem framing: for every GPS observation we generate K candidate arcs (road
segments) nearby; a classifier later scores each (observation, candidate) pair
("is this the arc the vehicle was really on?"), and an HMM/Viterbi decoder picks the
most coherent arc sequence.
"""
import numpy as np
import pandas as pd
import shapely
from shapely import STRtree


def arc_endpoints(nodes_xy, arcs):
    """Start/end coordinates (metres) of every arc. Returns A, B of shape (k,2)."""
    return nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]


def point_to_segment(P, A, B):
    """Vectorised point->segment geometry for paired rows of P, A, B (all (m,2)).

    Returns
      dist  : perpendicular/closest distance in metres
      frac  : position of the projection along the segment, clipped to [0,1]
      cross : signed lateral offset (>0 means point lies left of A->B direction)
    """
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-9)
    AP = P - A
    frac = np.clip((AP * AB).sum(1) / L2, 0.0, 1.0)
    closest = A + frac[:, None] * AB
    dist = np.linalg.norm(P - closest, axis=1)
    cross = (AB[:, 0] * AP[:, 1] - AB[:, 1] * AP[:, 0]) / np.sqrt(L2)
    return dist, frac, cross


def generate_candidates(P, A, B, radius=100.0, k=15):
    """Top-k nearest arcs per GPS point (all arcs within `radius` metres).

    If no arc is within `radius` of a point (GPS jump / gap), the single nearest
    arc is used so every observation has at least one candidate.

    Returns DataFrame [obs_idx, arc_id, dist, frac, cross, rank] sorted by obs then dist.
    """
    lines = shapely.linestrings(np.stack([A, B], axis=1))
    tree = STRtree(lines)
    pts = shapely.points(P)

    pairs = tree.query(pts, predicate="dwithin", distance=radius)  # (2, m)
    obs, arc = pairs[0], pairs[1]

    have = np.zeros(len(P), dtype=bool)
    have[np.unique(obs)] = True
    missing = np.where(~have)[0]
    if len(missing):
        near = tree.query_nearest(pts[missing], all_matches=False)  # (2, len(missing))
        obs = np.concatenate([obs, missing[near[0]]])
        arc = np.concatenate([arc, near[1]])

    dist, frac, cross = point_to_segment(P[obs], A[arc], B[arc])
    df = pd.DataFrame({"obs_idx": obs, "arc_id": arc, "dist": dist,
                       "frac": frac, "cross": cross})
    df = df.sort_values(["obs_idx", "dist"], kind="stable")
    df["rank"] = df.groupby("obs_idx").cumcount()
    df = df[df["rank"] < k].reset_index(drop=True)
    return df


def align_route_to_points(P, A, B, route, chunk=2000):
    """Assign each GPS point its reference arc from the ordered `.route`.

    The .route file only lists arcs in travel order, not which arc each GPS point
    belongs to. We recover per-point labels with a monotonic alignment (dynamic
    programming): choose route positions j_0 <= j_1 <= ... <= j_{n-1} minimising the
    total point-to-arc distance. The reference label of point t is route[j_t].

    Assumption (state it in the report): the vehicle never moves backwards along the
    reference route, and the route is treated as reference, not perfect truth.

    Returns (true_arc[n], route_pos[n], align_dist[n]).
    """
    n, r = len(P), len(route)
    Ar, Br = A[route], B[route]

    D = np.empty((n, r), dtype=np.float32)
    for s in range(0, n, chunk):
        p = P[s:s + chunk]
        m = len(p)
        d, _, _ = point_to_segment(np.repeat(p, r, axis=0),
                                   np.tile(Ar, (m, 1)), np.tile(Br, (m, 1)))
        D[s:s + m] = d.reshape(m, r)

    ar = np.arange(r)
    back = np.zeros((n, r), dtype=np.int32)
    F = D[0].astype(np.float64)
    for t in range(1, n):
        pm = np.minimum.accumulate(F)                      # best cost over j' <= j
        back[t] = np.maximum.accumulate(np.where(F == pm, ar, 0))  # argmin of prefix
        F = D[t] + pm
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(F))
    for t in range(n - 1, 0, -1):
        pos[t - 1] = back[t, pos[t]]
    return route[pos], pos, D[np.arange(n), pos]
