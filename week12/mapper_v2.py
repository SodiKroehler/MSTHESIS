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


# make mapping document
def get_full_ir_idx(row):
    ir_labels = {
        0: "Inspired",
        1: "Popular",
        2: "Moral",
        3: "Civic",
        4: "Economic",
        5: "Functional",
        6: "Ecological"
    }
    return f"{ir_labels.get(row['ir_idx'])}_{row['bucket']}"

fdf['full_ir_idx'] = fdf.apply(get_full_ir_idx, axis=1)
pvdf = fdf.pivot_table(index='clean', columns='full_ir_idx', values='logprob', aggfunc='first')
titles = fdf[['clean', 'title']].drop_duplicates()
pvdf = pvdf.reset_index().merge(titles, on='clean', how='left')
pvdf = pvdf.reset_index()
pvdf['idx'] = pvdf.index



# fdf['idx'] = fdf.index
# fdf = fdf[['idx', 'title', 'stance', 'clean', 'IR', 'ir_idx', 'bucket', 'score']]
# fdf.columns = ['idx', 'procon_subject', 'procon_stance', 'clean', 'IR', 'ir_idx', 'IR_bucket', 'score']


# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(fdf['clean'].astype(str))

# #i think with news we want this, but there's no reason to not also just use the procon title for now
# # clusterer = hdbscan.HDBSCAN(min_cluster_size=6)
# # labels = clusterer.fit_predict(X.toarray())
# # fdf['hdb_cluster'] = labels
# subjects = fdf['procon_subject'].unique().tolist()
# subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
# fdf['hdb_cluster'] = fdf['procon_subject'].map(subject_to_idx)

# # def get_x(row):
# #     z_score = round((row['score'] - 0.5) / 0.1)
# #     if row['IR_bucket'] == '-':
# #         z_score = 0-z_score
# #     return z_score

# def get_x(row):
#     z = round(row['score'])  # or scale if you want finer granularity
#     z = -z if row['IR_bucket'] == '-' else z

#     return z + 3  # Adjusting to fit the range of -3 to 3


# num_clusters = len(fdf['hdb_cluster'].unique())
# matrix = np.empty((7,num_clusters, 7), dtype=object)
# matrix.fill(None)

# def get_stance(row):
#     # ir_value = row['IR']
#     # idx = irs[irs['IR'] == ir_value]['idx'].values
#     # return idx[0] if len(idx) > 0 else None
#     return row['ir_idx']

# for i, row in fdf.iterrows():
#     y = row['hdb_cluster']
#     x = get_x(row)
#     z = get_stance(row)

#     #first we add it to the matrix
#     if matrix[x,y,z] is None:
#         matrix[x,y,z] = [row.idx]
#     elif matrix[x,y,z] == [-1]: #this is if we had seen a ghost peg
#         matrix[x,y,z] = [row.idx]
#     else:
#         matrix[x,y,z].append(row.idx)

#     #then we see if we need a ghost peg:
#     if matrix[(-x), y, z] is [] or matrix[(-x), y, z] is None:
#         matrix[(-x), y, z] = [-1]

#     # print(f"Processing row {i}: x={x}, y={y}, z={z}")
#     # print(f"Status of ghost peg at ({-x}, {y}, {z}): {matrix[(-x), y, z]}")

# # Save the matrix to a JSON file
# with open('procon_matrix_v1.json', 'w') as f:
#     json.dump(matrix.tolist(), f)

# fdf.to_csv('procon_coh_for_mapping.csv', index=False)


