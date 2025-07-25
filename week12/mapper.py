import json
import pandas as pd
# from sklearn.cluster import HDBSCAN
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

irs = pd.read_csv('interpretive_repertoires.csv')
irs = irs.drop_duplicates(subset=['IR']).reset_index(drop=True)
irs['ir_idx'] = irs.index
irs = irs[['IR', 'ir_idx']]


full_df = pd.read_csv('procon_coh_long.csv')

full_df['IR'] = full_df['IR'].astype(str)
irs['IR'] = irs['IR'].astype(str)
fdf = full_df.merge(irs, on='IR', how='left')
fdf = fdf.dropna(subset=['clean', 'logprob', 'IR'])
# scaler = MinMaxScaler()
# fdf['score'] = scaler.fit_transform(fdf[['logprob']])

scaler = StandardScaler()
fdf['score'] = scaler.fit_transform(fdf[['logprob']])


pvdf = fdf.pivot_table(index='clean', columns='ir_idx', values='score', aggfunc='first')
pvdf['idx'] = pvdf.index


want the fdf idx to be the unique idx of the clean text
entries = fdf['clean'].unique().tolist()
entries_to_idx = {entry: idx for idx, entry in enumerate(entries)}
fdf['idx'] = fdf['clean'].map(entries_to_idx)


fdf['idx'] = fdf.index
fdf = fdf[['idx', 'title', 'stance', 'clean', 'IR', 'ir_idx', 'bucket', 'score']]
fdf.columns = ['idx', 'procon_subject', 'procon_stance', 'clean', 'IR', 'ir_idx', 'IR_bucket', 'score']


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(fdf['clean'].astype(str))

#i think with news we want this, but there's no reason to not also just use the procon title for now
# clusterer = hdbscan.HDBSCAN(min_cluster_size=6)
# labels = clusterer.fit_predict(X.toarray())
# fdf['hdb_cluster'] = labels
subjects = fdf['procon_subject'].unique().tolist()
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
fdf['hdb_cluster'] = fdf['procon_subject'].map(subject_to_idx)

# def get_x(row):
#     z_score = round((row['score'] - 0.5) / 0.1)
#     if row['IR_bucket'] == '-':
#         z_score = 0-z_score
#     return z_score

def get_x(row):
    z = round(row['score'])  # or scale if you want finer granularity
    z = -z if row['IR_bucket'] == '-' else z

    return z + 3  # Adjusting to fit the range of -3 to 3


num_clusters = len(fdf['hdb_cluster'].unique())
matrix = np.empty((7,num_clusters, 7), dtype=object)
matrix.fill(None)

def get_stance(row):
    # ir_value = row['IR']
    # idx = irs[irs['IR'] == ir_value]['idx'].values
    # return idx[0] if len(idx) > 0 else None
    return row['ir_idx']

for i, row in fdf.iterrows():
    y = row['hdb_cluster']
    x = get_x(row)
    z = get_stance(row)

    #first we add it to the matrix
    if matrix[x,y,z] is None:
        matrix[x,y,z] = [row.idx]
    elif matrix[x,y,z] == [-1]: #this is if we had seen a ghost peg
        matrix[x,y,z] = [row.idx]
    else:
        matrix[x,y,z].append(row.idx)

    #then we see if we need a ghost peg:
    if matrix[(-x), y, z] is [] or matrix[(-x), y, z] is None:
        matrix[(-x), y, z] = [-1]

    # print(f"Processing row {i}: x={x}, y={y}, z={z}")
    # print(f"Status of ghost peg at ({-x}, {y}, {z}): {matrix[(-x), y, z]}")

# Save the matrix to a JSON file
with open('procon_matrix_v1.json', 'w') as f:
    json.dump(matrix.tolist(), f)

fdf.to_csv('procon_coh_for_mapping.csv', index=False)
