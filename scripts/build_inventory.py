from member_2_dhanush.preprocessing.inspect_dataset import inspect_partition
from pathlib import Path
Path('results/tables').mkdir(parents=True,exist_ok=True)
for p in ('train','student_test'):
    df=inspect_partition(p); df.to_csv(f'results/tables/{p}_inventory.csv',index=False)
    print(f'{p}: {len(df)} trips, {df.gps_rows.sum()} GPS observations')
