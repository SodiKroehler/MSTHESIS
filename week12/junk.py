import pandas as pd
import json
import nltk
from nltk.corpus import stopwords

df = pd.read_csv('~/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/combined_gdelt_ai.csv')
df = df[df['article_url'].str.contains('ai', case=False, na=False) & df['article_url'].str.contains('altman', case=False, na=False)]

# # df = pd.read_csv('procon_coh_for_mapping.csv')
# # # df = df[['hdb_cluster', 'procon_subject']]
# # # df.columns = ['topic', 'label']
# # # df.to_csv('topic_labels.csv', index=False)

# def get_x(row):
#     z = round(row['score'])  # or scale if you want finer granularity
#     z = -z if row['IR_bucket'] == '-' else z

#     return f"{z}: {row['score']}"#not updating this to +3 because we want to keep the range of -3 to 3 here

# # df['x'] = df.apply(get_x, axis=1)
# # df['z'] = df['ir_idx']
# # df['y'] = df['hdb_cluster'].astype(int)


# # with open("procon_matrix_v1.json", "r") as f:
# #     matrix = json.load(f)

# # df = pd.DataFrame(matrix)
# with open("procon_matrix_v1.json", "r") as f:
#     matrix = json.load(f)

# df = pd.read_csv("procon_coh_for_mapping.csv")
# df.set_index("idx", inplace=True)

# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))


# full_df = pd.read_csv("./../week11/procon_longer.csv")

# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     tokens = text.split()
#     tokens = [t for t in tokens if t not in stop_words]
#     return " ".join(tokens)



# def get_full_ir_idx(row):
#     ir_labels = {
#         0: "Inspired",
#         1: "Popular",
#         2: "Moral",
#         3: "Civic",
#         4: "Economic",
#         5: "Functional",
#         6: "Ecological"
#     }
#     return f"{ir_labels.get(row['ir_idx'])}_{row['IR_bucket']}"
# df['full_ir_idx'] = df.apply(get_full_ir_idx, axis=1)
# df['x'] = df.apply(get_x, axis=1)
# pdf = df.pivot_table(index=['procon_subject', 'clean'], columns='full_ir_idx', values='x', aggfunc='first')
# pdf = pdf.reset_index()

# # def topic_lookup(topic):
# #     topic_labels_df = pd.read_csv("topic_labels.csv", dtype={"topic": int, "label": str})
# #     d= dict(zip(topic_labels_df["topic"], topic_labels_df["label"]))
# #     return d.get(topic, f"Topic {topic}")

# # def ir_lookup(ir):


# # # Convert entries to DataFrame
# # entries_df = pd.DataFrame(entries, columns=["leaning", "topic_id", "ir_id", "ids"])
# # # pivot longer
# # df = entries_df.melt(id_vars=["leaning", "topic_id", "ir_id"], value_vars="ids", var_name="id_type", value_name="id")
# # df['leaning'] = df['leaning'].apply(lambda x: x - leaning_offset)
# # df['topic'] = df['topic_id'].apply(topic_lookup)
# # df['ir'] = df['ir_id'].apply(ir_lookup)


# full_df['raw_text'] = full_df['point'] + "\n" + full_df['explanation']
# full_df["clean"] = full_df["raw_text"].apply(clean_text)
# full_df = full_df[['raw_text', 'clean']]
# pdf = pd.merge(pdf, full_df, on="clean", how="left")


# pdf.to_csv("procon_mapped_entries.csv", index=False)