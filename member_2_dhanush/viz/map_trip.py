"""Interactive Folium map of one trip: raw GPS, reference route, nearby road network.

Usage (from repo root):
    python -m viz.map_trip "path/to/Student folder" train 00000003 [--out results/map_train_00000003.html]
Optional: --pred predicted_arcs.txt  (one arc_id per line) adds a 'Predicted route' layer
later, once the HMM output exists.
"""
import argparse
from pathlib import Path

import folium
import numpy as np

from preprocessing.loader import LocalProjector, load_trip


def arcs_to_lines(arc_ids, arcs, nodes):
    """List of [(lat, lon), (lat, lon)] polylines for the given arc ids."""
    a = arcs[arc_ids]
    return [[(nodes[i, 1], nodes[i, 0]), (nodes[j, 1], nodes[j, 0])] for i, j in a]


def make_map(d, pred_arcs=None, max_points=3000, max_net_arcs=6000, pad_deg=0.003):
    tr, nodes, arcs, route = d["track"], d["nodes"], d["arcs"], d["route"]
    lat_c, lon_c = tr[:, 1].mean(), tr[:, 0].mean()
    m = folium.Map(location=[lat_c, lon_c], zoom_start=15, tiles="OpenStreetMap")

    # road network inside the (padded) bounding box of the trajectory
    lo, hi = tr[:, :2].min(0) - pad_deg, tr[:, :2].max(0) + pad_deg
    inbox = ((nodes[:, 0] >= lo[0]) & (nodes[:, 0] <= hi[0]) &
             (nodes[:, 1] >= lo[1]) & (nodes[:, 1] <= hi[1]))
    sel = np.where(inbox[arcs[:, 0]] & inbox[arcs[:, 1]])[0]
    if len(sel) > max_net_arcs:
        sel = sel[np.linspace(0, len(sel) - 1, max_net_arcs).astype(int)]
    net = folium.FeatureGroup(name=f"Road network ({len(sel)} arcs)", show=True)
    folium.PolyLine(arcs_to_lines(sel, arcs, nodes), color="#888888", weight=2,
                    opacity=0.7).add_to(net)
    net.add_to(m)

    if route is not None:
        g = folium.FeatureGroup(name="Reference route (green)", show=True)
        folium.PolyLine(arcs_to_lines(route, arcs, nodes), color="green", weight=5,
                        opacity=0.85).add_to(g)
        g.add_to(m)
    if pred_arcs is not None:
        g = folium.FeatureGroup(name="Predicted route (red)", show=True)
        folium.PolyLine(arcs_to_lines(np.asarray(pred_arcs), arcs, nodes), color="red",
                        weight=4, opacity=0.85, dash_array="8").add_to(g)
        g.add_to(m)

    g = folium.FeatureGroup(name="Raw GPS (blue)", show=True)
    step = max(1, len(tr) // max_points)
    for i in range(0, len(tr), step):
        folium.CircleMarker((tr[i, 1], tr[i, 0]), radius=2, color="blue", fill=True,
                            fill_opacity=0.8, tooltip=f"obs {i}, t={tr[i, 2]:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker((tr[0, 1], tr[0, 0]), tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker((tr[-1, 1], tr[-1, 0]), tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)

    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:25px;left:25px;z-index:9999;background:white;"
              "padding:8px 12px;border:1px solid #888;font-size:13px'>"
              "<b>Legend</b><br><span style='color:blue'>●</span> raw GPS<br>"
              "<span style='color:green'>━</span> reference route<br>"
              "<span style='color:red'>╍</span> predicted route<br>"
              "<span style='color:#888'>━</span> road network</div>")
    m.get_root().html.add_child(folium.Element(legend))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("partition"); ap.add_argument("trip_id")
    ap.add_argument("--out", default=None); ap.add_argument("--pred", default=None)
    a = ap.parse_args()
    d = load_trip(Path(a.root) / a.partition / a.trip_id)
    pred = np.loadtxt(a.pred, dtype=int, ndmin=1) if a.pred else None
    out = Path(a.out or f"results/map_{a.partition}_{a.trip_id}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_map(d, pred).save(str(out))
    print("saved", out)
