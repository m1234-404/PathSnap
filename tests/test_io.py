from src.pathsnap.io import read_trip
from src.pathsnap.config import DATA_ROOT

def test_sample_trip_loads():
    x=read_trip(DATA_ROOT/'train'/'00000001')
    assert {'longitude','latitude','timestamp'} <= set(x['track'].columns)
    assert {'from_node_id','to_node_id'} <= set(x['arcs'].columns)
