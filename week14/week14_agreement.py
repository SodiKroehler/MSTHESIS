import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score


hdf = pd.read_csv('MSTHESIS_BACKEND_SHEET - PRODUCTION_jul16.csv')
gdf = pd.read_csv('set3_sampled_withgpt4o.csv')
hdf = hdf[hdf['rater_email'] == "sodikroehler@gmail.com"]
hdf.dropna(subset=['user_poli_leaning'], inplace=True)
gdf = gdf[['gkg_id', 'gpt_poli_leaning', 'gpt_article_subject', 'gpt_ai_leaning',
       'gpt_aireg_leaning', 'gpt_imm_leaning', 'gpt_poli_bias_amnt',
       'gpt_poli_bias_cause', 'gpt_stance_justification', 'gpt_poli_justification', 'source_leaning']]
df = pd.merge(hdf, gdf, on='gkg_id', how='left', suffixes=('_human', '_gpt4o'))
df['source_leaning'] = df['source_leaning'].str.upper()
# df.columns = ['gkg_id', 'timestamp', 'themes', 'DF_IDX', 'url', 'url_tuple',
#        'URL_new', 'images', 'authors', 'movies', 'tags', 'clean',
#        'gpt_leaning', 'LLM_RESPONSE', 'set_number', 'rater_email',
#        'coding_date', 'coding_time', 'old_bad_id', 'article_url', 'pull',
#        'source', 'title', 'text', 'date', 'human_clean_text',
#        'human_clean_title', 'ID', 'user_poli_leaning', 'user_agree_with',
#        'user_wants_to_see', 'user_article_subject', 'user_ai_leaning',
#        'user_aireg_leaning', 'user_imm_leaning', 'user_bias_amnt',
#        'user_bias_cause', 'set3_ID', 'gpt_poli_leaning', 'gpt_article_subject',
#        'gpt_ai_leaning', 'gpt_aireg_leaning', 'gpt_imm_leaning',
#        'gpt_poli_bias_amnt', 'gpt_poli_bias_cause', 'gpt_stance_justification',
#        'gpt_poli_justification']


def get_agreements(col1, col2):
    df[col1] = df[col1].fillna('UNSET')
    df[col2] = df[col2].fillna('UNSET')
    agreement_count = (df[col1] == df[col2]).sum()
    accuracy = accuracy_score(df[col1], df[col2])
    kappa = cohen_kappa_score(df[col1], df[col2])
    return col1, col2, agreement_count, accuracy, kappa

user_cols = [
    'user_poli_leaning',
    'user_poli_leaning',
    'gpt_poli_leaning',
    'user_article_subject',
    'user_ai_leaning',
    'user_aireg_leaning',
    'user_imm_leaning',
    'user_bias_amnt',
    'user_bias_cause'
]
gpt_cols = [
    'gpt_poli_leaning',
    'source_leaning',
    'source_leaning',
    'gpt_article_subject',
    'gpt_ai_leaning',
    'gpt_aireg_leaning',
    'gpt_imm_leaning',
    'gpt_poli_bias_amnt',
    'gpt_poli_bias_cause'
]
col1s = user_cols
col2s = gpt_cols
accuracy_data = [get_agreements(col1, col2) for col1, col2 in zip(col1s, col2s)]

total_count = len(df)
accuracy_table = pd.DataFrame(
    accuracy_data,
    columns=['Column 1', 'Column 2', 'Agreement Count', 'Accuracy', "Kappa"]
)
accuracy_table['Agreement Percentage'] = accuracy_table['Agreement Count'] / total_count
accuracy_table = accuracy_table[
    ['Column 1', 'Column 2', 'Agreement Count', 'Agreement Percentage', 'Accuracy', 'Kappa']
]

accuracy_table['Accuracy'] = accuracy_table['Accuracy'].apply(lambda x: f"{x:.3f}")
accuracy_table['Agreement Count'] = accuracy_table['Agreement Count'].apply(lambda x: f"{x}/{total_count}")
accuracy_table['Agreement Percentage'] = accuracy_table['Agreement Percentage'].apply(lambda x: f"{x:.3%}")
accuracy_table["Kappa"] = accuracy_table["Kappa"].apply(lambda x: f"{x:.3f}")

# print(accuracy_table)


#______________graphin________________
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Select the columns of interest
# pairs_to_plot = [
#     ('user_poli_leaning', 'gpt_poli_leaning'),
#     ('user_ai_leaning', 'gpt_ai_leaning'),
#     ('user_imm_leaning', 'gpt_imm_leaning')
# ]

# for user_col, gpt_col in pairs_to_plot:
#     disagreements = df[df[user_col] != df[gpt_col]].copy()
#     disagreements[[user_col, gpt_col]] = disagreements[[user_col, gpt_col]].fillna('UNSET')
#     disagreement_counts = disagreements.groupby([user_col, gpt_col]).size().reset_index(name='Count')
#     pivot_table = disagreement_counts.pivot(index=user_col, columns=gpt_col, values='Count').fillna(0)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='Blues')
#     plt.title(f'Disagreements: {user_col} vs {gpt_col}')
#     plt.xlabel('GPT Label')
#     plt.ylabel('User Label')
#     plt.tight_layout()
#     plt.show()


df_poli_disagree = df[df['user_poli_leaning'] != df['gpt_poli_leaning']]
df_ai_lean_disagree = df[df['user_ai_leaning'] != df['gpt_ai_leaning']]
df_imm_lean_disagree = df[df['user_imm_leaning'] != df['gpt_imm_leaning']]

df_poli_disagree.to_csv('disagreements_poli_leaning.csv', index=False)
df_ai_lean_disagree.to_csv('disagreements_ai_leaning.csv', index=False)
df_imm_lean_disagree.to_csv('disagreements_imm_leaning.csv', index=False)