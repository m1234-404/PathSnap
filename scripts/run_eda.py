import argparse
from src.pathsnap.config import DATA_ROOT
from src.pathsnap.io import read_trip
from member_4_pavithra.visualizations.folium_map import create_trip_map
p=argparse.ArgumentParser(); p.add_argument('--partition',default='train'); p.add_argument('--trip',required=True); a=p.parse_args()
x=read_trip(DATA_ROOT/a.partition/a.trip)
out=f'results/figures/{a.partition}_{a.trip}.html'
create_trip_map(x['track'],x['nodes'],x['arcs'],reference_arc_ids=x['route'],output=out)
print(out)
