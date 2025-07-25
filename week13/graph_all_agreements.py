import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score

# df_source = pd.read_csv('./full_study_files/set2_sampled_withgpt4o.csv')
df_source = pd.read_csv('full_study_files/pre_jul9_meeting_final.csv')
leaning_map = {
    "FAR_LEFT": "L",
    "LEFT": "L",
    "SLIGHT_LEFT": "L",
    "CENTER": "C",
    "SLIGHT_RIGHT": "R",
    "RIGHT": "R",
    "FAR_RIGHT": "R",
    "NONE": "N",
    "left": "L",
    "right": "R",
    "center": "C",
    "none": "N",
    "global": "N",
    '': 'N'
}
df_source['source_lean'] = df_source['source_leaning'].map(leaning_map)
df_source['gpt_lean'] = df_source['gpt_leaning'].map(leaning_map)
df_source['human_lean'] = df_source['human_leaning'].map(leaning_map)

adf = df_source[['source_lean', 'gpt_lean', 'human_lean']].dropna().copy()



# Calculate pairwise accuracies, agreement counts, and Cohen's kappa
accuracy_data = [
    [
        'human_lean',
        'gpt_lean',
        accuracy_score(adf['human_lean'], adf['gpt_lean']),
        (adf['human_lean'] == adf['gpt_lean']).sum(),
        cohen_kappa_score(adf['human_lean'], adf['gpt_lean'])
    ],
    [
        'human_lean',
        'source_lean',
        accuracy_score(adf['human_lean'], adf['source_lean']),
        (adf['human_lean'] == adf['source_lean']).sum(),
        cohen_kappa_score(adf['human_lean'], adf['source_lean'])
    ],
    [
        'gpt_lean',
        'source_lean',
        accuracy_score(adf['gpt_lean'], adf['source_lean']),
        (adf['gpt_lean'] == adf['source_lean']).sum(),
        cohen_kappa_score(adf['gpt_lean'], adf['source_lean'])
    ]
]

total_count = len(adf)
accuracy_table = pd.DataFrame(
    accuracy_data,
    columns=['Column 1', 'Column 2', 'Accuracy', 'Agreement Count', "Kappa"]
)

# Format accuracy, agreement count, and kappa
accuracy_table['Accuracy'] = accuracy_table['Accuracy'].apply(lambda x: f"{x:.3f}")
accuracy_table['Agreement Count'] = accuracy_table['Agreement Count'].apply(lambda x: f"{x}/{total_count}")
accuracy_table["Kappa"] = accuracy_table["Kappa"].apply(lambda x: f"{x:.3f}")

print("\nPairwise Accuracy Table:")
print(accuracy_table)
