from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


sdf = pd.read_csv("final_pca_and_gpt_framing.csv")

y_labels = {
    1: "Political",
    2: "Public Sentiment",
    3: "Cultural Identity",
    4: "Morality and Ethics",
    5: "Fairness and Equality",
    6: "Legality, Constitutionality, Jurisdiction",
    7: "Crime and Punishment",
    8: "Security and Defense",
    9: "Health and Safety",
    10: "Quality of Life",
    11: "Economics",
    12: "Capacity and Resources",
    13: "Policy Description, Prescription, Evaluation",
    14: "External Regulation and Reputation",
    15: "Other"
}


x_axis_labels = {
    1: ("Left", "Right"),
    2: ("Disapproval", "Support"),
    3: ("Assimilation", "Tradition"),
    4: ("Secular Values", "Religious Values"),
    5: ("Discrimination", "Equity"),
    6: ("Unlawful", "Constitutional"),
    7: ("Leniency", "Retribution"),
    8: ("Vulnerable", "Protected"),
    9: ("Harmful", "Safe"),
    10: ("Discomfort", "Wellbeing"),
    11: ("Cost", "Benefit"),
    12: ("Insufficient", "Sufficient"),
    13: ("Ineffective", "Effective"),
    14: ("Criticized", "Respected"),
    15: ("Undefined", "Undefined")
}

cmap = plt.cm.get_cmap("tab20", len(y_labels))
colors = {y: cmap(i) for i, y in enumerate(y_labels)}

sdf.dropna(subset=['gpt_leaning'], inplace=True)

sdf["gpt_leaning"] = sdf["gpt_leaning"].replace('Undefined', 0.0)
sdf['x'] = sdf['gpt_leaning'].astype(float)
sdf['y'] = sdf['y'].astype(float)


fig, ax = plt.subplots(figsize=(12, 10))

patches = []
for y in y_labels:
    color = colors[y]
    ax.axhspan(y - 0.5, y + 0.5, facecolor=color, alpha=0.2)
    patches.append(mpatches.Patch(color=color, label=y_labels[y]))


for x, item in sdf.iterrows():
    x = item['x']
    y = item['y']
    label = item['frame']
    ax.scatter(x, y, color='blue', alpha=0.1)
    # ax.text(x + 0.1, y, label, verticalalignment='center', fontsize=9)
    # ax.text(x + 0.1, y, f"{x:.2f}", va='center', fontsize=9)


ax.set_xlim(-6, 6)
ax.set_ylim(0.5, 15.5)
ax.set_xlabel("Leaning")
# ax.set_ylabel("Frame")    
ax.set_yticks([])  

for y, (left, right) in x_axis_labels.items():
    ax.text(-6, y, left, va='center', ha='right', fontsize=8, color='gray')
    ax.text(6, y, right, va='center', ha='left', fontsize=8, color='gray')


ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

ax.legend(handles=patches[::-1], bbox_to_anchor=(1.09, 1), loc='upper left', borderaxespad=0.)
ax.set_title("Framing Variation")
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()