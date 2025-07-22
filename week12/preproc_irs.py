import ast
import pandas as pd

with open("interpretive_reps2.json", "r") as f:
    raw = f.read()

irs_data = ast.literal_eval(raw)
idf = pd.DataFrame(irs_data)
idf.to_csv("interpretive_repertoires.csv", index=False)


# Define a simple clean_text function (customize as needed)
def clean_text(text):
    return text.strip()

# Use idf as full_df
full_df = idf

full_df['raw_text'] = full_df['point'] + "\n" + full_df['explanation']
full_df["clean"] = full_df["raw_text"].apply(clean_text)



full_df = full_df.drop(columns=['likes', 'dislikes'])