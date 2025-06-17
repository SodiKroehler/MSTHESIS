import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import random

y_labels = {
    1: "Political Factors and Implications",
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

fig, ax = plt.subplots(figsize=(12, 10))
# fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)

patches = []
for y in y_labels:
    color = colors[y]
    ax.axhspan(y - 0.5, y + 0.5, facecolor=color, alpha=0.2)
    patches.append(mpatches.Patch(color=color, label=y_labels[y]))

# Plot data points and annotate with x value
# for item in data:
#     x = item['x']
#     y = item['y']
#     ax.scatter(x, y, color='blue')
#     ax.text(x + 0.1, y, f"{x:.2f}", va='center', fontsize=9)

sdf = pd.read_csv("procon_dev_withChatGPTFrames_for_graphing.csv")
# sdf['y_offset'] = np.random.uniform(0.5, 15.5, sdf.shape[0])


for x, item in sdf.iterrows():
    x = item['y_offset']

    y = item['frame_code']
    label = y_labels[int(y)] if y in y_labels else "Other"
    ax.scatter(x, y, color='blue')
    # ax.text(x + 0.1, y, label, verticalalignment='center', fontsize=9)
    ax.text(x + 0.1, y, f"{x:.2f}", va='center', fontsize=9)


# Axis formatting
ax.set_xlim(-6, 6)
ax.set_ylim(0.5, 15.5)
ax.set_xlabel("Leaning")
ax.set_ylabel("Frame")
ax.set_yticks([])  # remove default y-ticks

# Add left and right pole labels
# for y, (left, right) in x_axis_labels.items():
#     ax.text(-6, y, left, va='center', ha='right', fontsize=8, color='gray')
#     ax.text(6, y, right, va='center', ha='left', fontsize=8, color='gray')

# Add vertical center line
ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# Add legend
ax.legend(handles=patches, bbox_to_anchor=(1.09, 1), loc='upper left', borderaxespad=0.)

# Title and grid
ax.set_title("Framing Variation")
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
