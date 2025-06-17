from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tqdm
import numpy as np
import pandas as pd

def get_frame_code_from_frame(text):

    y_labels_semantic_order = {
            1.0: "Political",
            2.0: "Public Sentiment",
            3.0: "Cultural Identity",
            4.0: "Morality and Ethics",
            5.0: "Fairness and Equality",
            6.0: "Legality, Constitutionality, Jurisdiction",
            7.0: "Crime and Punishment",
            8.0: "Security and Defense",
            9.0: "Health and Safety",
            10.0: "Quality of Life",
            11.0: "Economics",
            12.0: "Capacity and Resources",
            13.0: "Policy Description, Prescription, Evaluation",
            14.0: "External Regulation and Reputation",
            15.0: "Other"
        }

    for real in y_labels_semantic_order:
        if text.title() == y_labels_semantic_order[real]:
            return real

    return -1.0

idf = pd.read_csv('procon_dev_withChatGPTFrames.csv')
pca_model = SentenceTransformer("all-MiniLM-L6-v2") 
idf['embedding'] = idf['a'].apply(lambda x: pca_model.encode(x))
idf['frame_code'] = idf.apply(lambda row: get_frame_code_from_frame(row['gpt_frame']), axis=1)
idf["y_offset"] = np.nan

# Loop through groups and compute offsets
for code, group in idf.groupby("gpt_frame"):
    try:
        embeddings = np.vstack(group["embedding"].values)

        if len(embeddings) > 1:
            pca = PCA(n_components=1)
            reduced = pca.fit_transform(embeddings)

            scaler = MinMaxScaler(feature_range=(-5, 5))
            normalized = scaler.fit_transform(reduced).flatten()
        else:
            normalized = np.array([0.0])  # Single entry gets 0

        idf.loc[group.index, "y_offset"] = normalized

    except Exception as e:
        print(f"Skipping group {code} due to error: {e}")
        continue

idf.drop(columns=["embedding"], inplace=True)
idf['y'] = idf.apply(lambda row: float(row['frame_code']) + float(row['y_offset']), axis=1)
idf.to_csv("procon_dev_withChatGPTFrames_for_graphing.csv", index=False)