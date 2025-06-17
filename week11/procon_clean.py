import json
import random
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


with open('./../raw/PROCON/procon_full_may2025.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)

rows = []
for _, row in df.iterrows():
    base = row.drop(['pro_arguments', 'con_arguments']).to_dict()
    for arg in row.get('pro_arguments', []):
        new_row = base.copy()
        new_row['stance'] = 'pro'
        new_row['point'] = arg.get('point')
        new_row['explanation'] = '\n'.join(arg.get('explanation', []))
        rows.append(new_row)
    for arg in row.get('con_arguments', []):
        new_row = base.copy()
        new_row['stance'] = 'con'
        new_row['point'] = arg.get('point')
        new_row['explanation'] = '\n'.join(arg.get('explanation', []))
        rows.append(new_row)
df_long = pd.DataFrame(rows)
df_long.to_csv('./procon_longer.csv')