from pathlib import Path
import pandas as pd

def read_trip(trip_dir: str | Path, include_route: bool = True) -> dict:
    """Read one trip. Set include_route=False in all prediction-serving code."""
    d = Path(trip_dir)
    trip_id = d.name
    track = pd.read_csv(d / f"{trip_id}.track", sep="\t", header=None,
                        names=["longitude", "latitude", "timestamp"])
    nodes = pd.read_csv(d / f"{trip_id}.nodes", sep="\t", header=None,
                        names=["longitude", "latitude"])
    arcs = pd.read_csv(d / f"{trip_id}.arcs", sep="\t", header=None,
                       names=["from_node_id", "to_node_id"], dtype="int64")
    route_path = d / f"{trip_id}.route"
    route = None
    if include_route and route_path.exists():
        route = pd.read_csv(route_path, sep="\t", header=None,
                            names=["arc_id"], dtype="int64").arc_id.tolist()
    return {"trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}
