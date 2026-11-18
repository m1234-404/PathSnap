"""Raw-file loading and metric projection for PathSnap trips.

Raw formats (no headers, whitespace separated, .track has CRLF endings):
  .track : lon, lat, timestamp(s)
  .nodes : lon, lat            (row index = node_id)
  .arcs  : from_node, to_node  (row index = arc_id)
  .route : arc_id              (reference route, chronological; train/student only)
"""
from pathlib import Path

import numpy as np

EARTH_R = 6371008.8  # metres


def load_trip(trip_dir):
    """Load one trip folder into a dict of numpy arrays."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    d = {"trip_id": tid}
    d["track"] = np.loadtxt(trip_dir / f"{tid}.track", ndmin=2)                # (n,3)
    d["nodes"] = np.loadtxt(trip_dir / f"{tid}.nodes", ndmin=2)                # (m,2)
    d["arcs"] = np.loadtxt(trip_dir / f"{tid}.arcs", dtype=np.int64, ndmin=2)  # (k,2)
    rf = trip_dir / f"{tid}.route"
    d["route"] = np.loadtxt(rf, dtype=np.int64, ndmin=1) if rf.exists() else None
    return d


class LocalProjector:
    """Local equirectangular projection (lon/lat degrees -> metres).

    Why: GPS/nodes are angular coordinates, so Euclidean distance in degrees is
    meaningless. Each trip covers at most a few tens of km, so a plane tangent at the
    trip's mean position gives sub-percent distance error, and is fast and invertible.
    Nodes and GPS points of a trip are always projected with the SAME projector.
    """

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = np.deg2rad(1.0) * EARTH_R * np.cos(np.deg2rad(self.lat0))
        self.ky = np.deg2rad(1.0) * EARTH_R

    @classmethod
    def from_track(cls, track):
        return cls(track[:, 0].mean(), track[:, 1].mean())

    def forward(self, lon, lat):
        return np.column_stack([(np.asarray(lon) - self.lon0) * self.kx,
                                (np.asarray(lat) - self.lat0) * self.ky])

    def inverse(self, xy):
        xy = np.asarray(xy)
        return xy[:, 0] / self.kx + self.lon0, xy[:, 1] / self.ky + self.lat0
