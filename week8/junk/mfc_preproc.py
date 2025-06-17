#read in json

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from openai import OpenAI


# import pandas as pd
# import openai
import time

import pandas as pd
import json

OPENAI_API_KEY = 'nope'
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

# old_frames_dict = {}

# with open('./media_frames_corpus/annotations/codes.json') as f:
#     old_frames_dict = json.load(f)
# old_frames = {v.upper(): float(k) for k, v in old_frames_dict.items()}

# # idf = pd.read_csv("../raw/PFC/harvard thing/training_data_frames_final.csv")
# idf = pd.read_csv("./with_real_and_gpt_frames.csv")
# frame_mappings_df = pd.read_csv("/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week8/label_matches.csv")

# new_frames ={}
# for index, row in frame_mappings_df.iterrows():
#     if row['Original'].upper() in old_frames:
#         new_frame = row['Boydston'].upper()
#         new_frame_key = None
#         for key, value in y_labels_semantic_order.items():
#             if value.upper() == new_frame:
#                 new_frame_key = key
#                 break
#         new_frames[row['Original'].upper()] = new_frame_key


# idf.dropna(subset=['text'], inplace=True)

# def get_frame(h_label):
#     h_label = str(h_label)

#     if not h_label:
#         return None
#     if h_label.upper() in new_frames:
#         return old_frames[h_label.upper()]
#     elif h_label.upper() in old_frames and h_label.upper() not in new_frames:
#         print(f"Old frame: {h_label.upper()}")
#         return None
#     else:
#         return None
    
# idf['frame'] = idf['frame'].apply(lambda x: x.upper() if x.upper() in old_frames else None)
# idf['frame_code'] = idf['frame'].apply(get_frame)

# idf['new_frame_code'] = idf['frame'].apply(get_frame)
# idf.to_csv("with_new_real_and_gpt_frames.csv", index=False)

# idf.dropna(subset=['text'], inplace=True)
# idf = idf[idf['text'].str.len() > 0]

# model = SentenceTransformer("all-MiniLM-L6-v2") 
# idf['embedding'] = idf['text'].apply(lambda x: model.encode(x))

# gpt_df = pd.read_csv("idf_withgpt.csv")

# for ii, irow in idf.iterrows():
#     for gi, grow in gpt_df.iterrows():
#         if grow['text'] == irow['text']:
#             idf.at[ii, 'gpt_frame'] = grow['frame']
#             idf.at[ii, 'gpt_leaning'] = grow['leaning']
#             break

# # idf['leaning'] = idf['leaning'].astype(float)
# # idf['gpt_frame'] = idf['frame'].astype(float)


# idf.to_csv("with_real_and_gpt_frames.csv", index=False)

# def build_chat_gpt():
#     # chatgpt
#     y_labels_semantic_order = {
#         1.0: "Political",
#         2.0: "Public Sentiment",
#         3.0: "Cultural Identity",
#         4.0: "Morality and Ethics",
#         5.0: "Fairness and Equality",
#         6.0: "Legality, Constitutionality, Jurisdiction",
#         7.0: "Crime and Punishment",
#         8.0: "Security and Defense",
#         9.0: "Health and Safety",
#         10.0: "Quality of Life",
#         11.0: "Economics",
#         12.0: "Capacity and Resources",
#         13.0: "Policy Description, Prescription, Evaluation",
#         14.0: "External Regulation and Reputation",
#         15.0: "Other"
#     }
#     ambivalence_labels = {
#         1.0: ("Left", "Right"),
#         2.0: ("Disapproval", "Support"),
#         3.0: ("Assimilation", "Tradition"),
#         4.0: ("Secular Values", "Religious Values"),
#         5.0: ("Discrimination", "Equity"),
#         6.0: ("Unlawful", "Constitutional"),
#         7.0: ("Leniency", "Retribution"),
#         8.0: ("Vulnerable", "Protected"),
#         9.0: ("Harmful", "Safe"),
#         10.0: ("Discomfort", "Wellbeing"),
#         11.0: ("Cost", "Benefit"),
#         12.0: ("Insufficient", "Sufficient"),
#         13.0: ("Ineffective", "Effective"),
#         14.0: ("Criticized", "Respected"),
#         15.0: ("Undefined", "Undefined")
#     }

    

#     # sdf = idf.sample(5)
#     sdf = idf
#     sdf['id'] = sdf.index

#     df = sdf[['id', 'text']]

#     # Parameters
#     BATCH_SIZE = 10  # Number of rows per API call

#     PROMPT_TEMPLATE = """
#     You will be given a list of numbered texts. For each one, identify:
#     1. The most relevant *frame* from the list of 15 below.
#     2. The *leaning* within that frame, where:
#     - Leaning is anything from -5.0 to 5.0, depending on which side best represents the stance taken in the text.
#     - Only assign a leaning for the chosen frame.
#     3. If the text does not fit any of the frames, return Other for frame and Undefined for leaning.
#     4. Be as specific as possible, try to output floating point numbers instead of integers.

#     Return your result as a JSON array, one object per text, like this:
#     [{{"frame": 1, "leaning": 2}}, ...]

#     Frames:
#     {frames}

#     Leanings:
#     Each frame has its own leaning scale:
#     {leanings}

#     Texts:
#     {texts}

#     ONLY return the JSON array. Do not include any other explanation.
#     """
#     client = OpenAI(api_key=OPENAI_API_KEY)
#     def call_chatgpt_batch(texts):
#         # Construct the prompt with all texts for this batch
#         joined_texts = "\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
#         frames = json.dumps(y_labels_semantic_order)
#         leanings = json.dumps(ambivalence_labels)
#         prompt = PROMPT_TEMPLATE.format(texts=joined_texts, frames = frames, leanings=leanings)


#         try:

#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {"role": "system", "content": "You are a journalist who analyzes frames in media articles."},
#                     {"role": "user", "content": prompt}],
#                 max_tokens=1000,
#                 temperature=0
#             )

#             # output = response['choices'][0]['message']['content'].strip()
#             output = response.choices[0].message.content.strip()
#             output = output.strip('`').split('\n', 1)[-1].rsplit('\n', 1)[0]
#             # Parse the response as a JSON array
#             json_output = json.loads(output)
#             frames, leanings = [], []
#             # Extract frames and leanings from the JSON output
#             for item in json_output:
#                 frame = item.get('frame', '')
#                 leaning = item.get('leaning', '')
#                 frames.append(frame)
#                 leanings.append(leaning)


#             return frames, leanings

            
#         # return scores

#         except Exception as e:
#             print(f"Error: {e}")
#             return [None] * len(texts), [None] * len(texts)

#     # Score storage
#     all_frames = []
#     all_leanings = []

#     # Process in batches
#     for i in range(0, len(df), BATCH_SIZE):
#         batch = df.iloc[i:i+BATCH_SIZE]
#         texts = batch["text"].tolist()
#         frame, leaning = call_chatgpt_batch(texts)
#         all_frames.extend(frame)
#         all_leanings.extend(leaning)

#         # Be nice to the API
#         time.sleep(1)

#     # Add scores to DataFrame
#     df["frame"] = all_frames
#     df["leaning"] = all_leanings



#     df.to_csv("idf_withgpt.csv", index=False)


