import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import re
from sentence_transformers import SentenceTransformer, util
import hdbscan
from sklearn.cluster import KMeans
import random
from charm import CHARM
import json
from sentence_transformers import SentenceTransformer, util

import time

with open('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/from_crc/ir_ourCategories_mapping.json', 'r') as f:
    ir_custom_scores = json.load(f)


df = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/from_crc/cohcharm_base_file.csv')

ir_labels = {
    0: "Inspired",
    1: "Popular",
    2: "Moral",
    3: "Civic",
    4: "Economic",
    5: "Functional",
    6: "Ecological"
}

leanings_maps = {
    "FAR_LEFT": -3,
    "LEFT": -2,
    "SLIGHT_LEFT": -1,
    "CENTER": 0,
    "SLIGHT_RIGHT": 1,
    "RIGHT": 2,
    "FAR_RIGHT": 3,
    "NONE": None,
    "UNDEFINED": None,
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "OPTIMISTIC": 1,
    "PESSIMISTIC": -1,
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "PRO_REGULATION": 1,
    "ANTI_REGULATION": -1,
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "PERMISSIVE": 1,
    "STRICT": -1,
}

for ir in ir_custom_scores.keys():
    for bucket in ["+", "-"]:
        for stance in ir_custom_scores[ir][bucket].keys():
            if ir_custom_scores[ir][bucket][stance] in leanings_maps:
                newval = leanings_maps[ir_custom_scores[ir][bucket][stance]] + random.uniform(0, 0.0001)  # add a small random value to avoid zeros
                ir_custom_scores[ir][bucket][stance] = newval
            else:
                raise ValueError(f"Unknown leaning value: {ir_custom_scores[ir][bucket][stance]} for {ir} {bucket} {stance}")

stances = [
    "poli",
    "ai",
    "aireg",
    "imm"
]

model = CHARM(ir_custom_scores, ir_labels, stances)
# df = df.head(2)
start_time = time.time()
#make sure we can do it here first
enc_model = SentenceTransformer('all-MiniLM-L6-v2')

edf = df.copy()
edf = df.drop_duplicates(subset=['A', 'P'], keep='first')[['A', 'P', 'p_text']]
edf["embeddings"] = edf["p_text"].apply(lambda x: enc_model.encode(edf["p_text"].tolist()))

edf.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/from_crc/cohcharm_base_file_with_embeddings.csv', index=False)


charm_df, charm_mat = model.make_mat(df, edf)

df = df.merge(charm_df, on='A', how='left')
df.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/from_crc/simplecharm_full_jul31.csv', index=False)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

#2 rows took 0.9563069343566895 with the updated method
#we know its not anymore, but worse case was 30000 rows, so 30000 * 0.9563069343566895 = 28,689 seconds = 7.9 hours we're still safe. so lets run it.