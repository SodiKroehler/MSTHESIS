import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer


ldf = pd.read_csv("./with_new_real_and_gpt_frames.csv")

model = SentenceTransformer("all-MiniLM-L6-v2") 
ldf['embedding'] = ldf['text'].apply(lambda x: model.encode(x))

# offsets = []
ldf["relative_offset"] = np.nan

# Loop through groups and compute offsets
for code, group in ldf.groupby("new_frame_code"):
    try:
        embeddings = np.vstack(group["embedding"].values)

        if len(embeddings) > 1:
            pca = PCA(n_components=1)
            reduced = pca.fit_transform(embeddings)

            scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
            normalized = scaler.fit_transform(reduced).flatten()
        else:
            normalized = np.array([0.0])  # Single entry gets 0

        # Assign to the correct indices
        ldf.loc[group.index, "relative_offset"] = normalized

    except Exception as e:
        print(f"Skipping group {code} due to error: {e}")
        continue

# # Reorder and insert into the DataFrame
# offsets.sort()  # sort by original index
# ldf["relative_offset"] = [val for idx, val in offsets]
ldf.drop(columns=["embedding"], inplace=True)
ldf.to_csv("final_pca_and_gpt_framing.csv", index=False)