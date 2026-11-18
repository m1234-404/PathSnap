from pathlib import Path
import pandas as pd
from src.pathsnap.config import DATA_ROOT
from src.pathsnap.io import read_trip

def inspect_partition(partition: str) -> pd.DataFrame:
    rows=[]
    for d in sorted((DATA_ROOT / partition).iterdir()):
        if not d.is_dir(): continue
        x=read_trip(d); t=x['track']; n=x['nodes']; a=x['arcs']
        dt=t.timestamp.diff()
        rows.append(dict(partition=partition, trip_id=x['trip_id'],
            gps_rows=len(t), node_rows=len(n), arc_rows=len(a),
            route_rows=len(x['route']) if x['route'] is not None else None,
            missing_gps=int(t.isna().any(axis=1).sum()),
            non_monotonic_timestamps=int((dt.dropna()<=0).sum()),
            median_time_gap=float(dt.median()) if len(dt.dropna()) else None,
            max_time_gap=float(dt.max()) if len(dt.dropna()) else None,
            min_lon=float(t.longitude.min()), max_lon=float(t.longitude.max()),
            min_lat=float(t.latitude.min()), max_lat=float(t.latitude.max())))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    for part in ('train','student_test'):
        out=inspect_partition(part)
        out.to_csv(f'results/tables/{part}_inventory.csv',index=False)
        print(part, len(out), 'trips')
