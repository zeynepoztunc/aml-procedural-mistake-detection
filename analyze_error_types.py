import pandas as pd
import json

splits = json.load(open('er_annotations/recordings_combined_splits.json'))
df = pd.read_csv('annotations/annotation_csv/error_annotations.csv')

error_types = [
    'Preparation Error',
    'Measurement Error',
    'Order Error',
    'Timing Error',
    'Technique Error',
    'Temperature Error',
    'Missing Step',
    'Other'
]

for split in ['train', 'val', 'test']:
    print(f'--- {split.upper()} ---')
    ids = splits[split]
    sdf = df[df['recording_id'].isin(ids)]
    for col in error_types:
        print(f'{col}:', sdf[col].astype(int).sum())
    print()
