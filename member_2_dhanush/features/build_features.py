import numpy as np
import pandas as pd

def add_observation_features(track: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Attach observation-level motion and time features to candidate rows."""
    t=track.copy()
    t['dt_s']=t.timestamp.diff().fillna(1).clip(lower=1e-3)
    lat1=np.deg2rad(t.latitude); lon1=np.deg2rad(t.longitude)
    dlat=lat1.diff(); dlon=lon1.diff()
    h=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat1.shift())*np.sin(dlon/2)**2
    t['gps_step_m']=(6371000*2*np.arcsin(np.sqrt(h.clip(0,1)))).fillna(0)
    t['gps_speed_mps']=t.gps_step_m/t.dt_s
    t['bearing_rad']=np.arctan2(dlon*np.cos(lat1),dlat).fillna(0)
    out=candidates.merge(t.reset_index(names='observation_id'),on='observation_id',how='left')
    # TODO: add point-to-road-segment distance, heading alignment, network
    # connectivity, candidate rank, and temporal context; document all units.
    return out.replace([np.inf,-np.inf],np.nan).fillna(0)
