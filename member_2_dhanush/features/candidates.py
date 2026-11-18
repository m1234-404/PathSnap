import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def generate_candidates(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame,
                        k_nodes: int = 8) -> pd.DataFrame:
    """Starter: nearest-node candidates mapped to incident arcs.

    TODO (Dhanush): replace/justify candidate generation using point-to-segment
    distance in a suitable metric CRS or geodesic calculation. The nearest-node
    approximation is only a baseline, not a final map-matcher.
    """
    # Local equirectangular approximation around this trip's mean latitude (metres).
    lat0=np.deg2rad(track.latitude.mean())
    def xy(df):
        return np.c_[np.deg2rad(df.longitude)*6371000*np.cos(lat0),
                     np.deg2rad(df.latitude)*6371000]
    tree=cKDTree(xy(nodes))
    node_to_arcs={}
    for arc_id, row in arcs.iterrows():
        node_to_arcs.setdefault(int(row.from_node_id),[]).append(int(arc_id))
        node_to_arcs.setdefault(int(row.to_node_id),[]).append(int(arc_id))
    rows=[]
    dist, idx=tree.query(xy(track),k=min(k_nodes,len(nodes)))
    if idx.ndim==1: idx=idx[:,None]; dist=dist[:,None]
    for obs_id in range(len(track)):
        seen=set()
        for d,nid in zip(dist[obs_id],idx[obs_id]):
            for aid in node_to_arcs.get(int(nid),[]):
                if aid in seen: continue
                seen.add(aid)
                rows.append(dict(observation_id=obs_id,candidate_arc_id=aid,
                                 nearest_node_distance_m=float(d)))
    return pd.DataFrame(rows)
