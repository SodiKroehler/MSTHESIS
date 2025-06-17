from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


frames = [
    "Economic", "Capacity and resources", "Morality", "Fairness and Equality", 
    "Legality, Constitutionality and Jurisprudence", "Policy Prescription and Evaluation", 
    "Crime and Punishment", "Security and defense", "Health and safety", "Quality of life", 
    "Cultural identity", "Public opinion", "External Regulation and Reputation", 
    "Other", "Political"
]
frames = [x.upper() for x in frames]

# idf = pd.read_csv("../raw/PFC/harvard thing/training_data_frames_final.csv")



# def get_frame(h_label):
#     # uframes = [x.upper() for x in frames]
#     if h_label.upper() in frames:
#         return h_label.upper()
#     else:
#         return ""

# idf['flabel'] = idf['frame'].apply(get_frame)
# idf.dropna(subset=['text'], inplace=True)
# idf = idf[idf['text'].str.len() > 0]

# dummy_data = []

# model = SentenceTransformer("all-MiniLM-L6-v2") 

# # for index, row in idf.iterrows():
# #     text = row['text']
# #     embedding = model.encode(text)
# #     frame = get_frame(row['frame'] )

# ldf = pd.DataFrame(columns=idf.columns)

# for frame in frames:
#     f_idf = idf[idf['flabel'] == frame]
#     # print(f"Frame: {frame} has {len(f_idf)} samples")
#     if len(f_idf) < 10:
#         ldf = pd.concat([ldf, f_idf])
#     else:
#         f_idf = f_idf.sample(n=10, random_state=42)
#         ldf = pd.concat([ldf, f_idf]) #ldf.append(f_idf)


#     # for i in range(7):  # 3 samples per frame

#     #     dummy_data.append({"frame": frame, **{f"dim_{j}": embedding[j] for j in range(10)}})


# ldf['embedding'] = ldf['text'].apply(lambda x: model.encode(x))

# # # Separate features and labels
# # X = df[[f"dim_{j}" for j in range(10)]].values
# X = np.array([np.array(x) for x in ldf['embedding']])
# y = ldf["flabel"].values

# # # Standardize features before PCA
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # # Apply PCA to reduce to 2D
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# # # # Plot the 2D PCA
# # plt.figure(figsize=(12, 8))
# # for frame in set(y):
# #     indices = [i for i, label in enumerate(y) if label == frame]
# #     plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=frame)

# # plt.title("PCA of Frame Embeddings")
# # plt.xlabel("Principal Component 1")
# # plt.ylabel("Principal Component 2")
# # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()



# ldf["pc1"] = X_pca[:, 0]
# ldf["pc2"] = X_pca[:, 1]

# # Plot convex hulls per frame
# plt.figure(figsize=(12, 8))
# colors = plt.cm.tab20.colors

# for i, frame in enumerate(ldf["frame"].unique()):
#     points = ldf[ldf["frame"] == frame][["pc1", "pc2"]].values
#     if len(points) >= 3:  # Convex hull needs at least 3 points
#         hull = ConvexHull(points)
#         plt.fill(points[hull.vertices, 0], points[hull.vertices, 1],
#                  alpha=0.3, label=frame, color=colors[i % len(colors)])
#     else:
#         # If not enough points for a hull, plot a point instead
#         plt.scatter(points[:, 0], points[:, 1], label=frame, color=colors[i % len(colors)])

# plt.title("Frame Clusters in PCA Space (Convex Hulls)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
# plt.grid(True)
# plt.tight_layout()
# plt.show()



sdf = pd.read_csv("../raw/PFC/harvard thing/training_data_scored.csv")


# # # Plot the 2D PCA
# plt.figure(figsize=(12, 8))

# plt.scatter(sdf['leaning'], sdf['frame'])

# plt.title("PCA of Frame Embeddings")
# plt.xlabel("Leaning")
# plt.ylabel("Frame")
# # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# plt.grid(True)
# plt.tight_layout()

# plt.show()

# y_labels = {
#     1: "Political Factors and Implications",
#     2: "Public Sentiment",
#     3: "Cultural Identity",
#     4: "Morality and Ethics",
#     5: "Fairness and Equality",
#     6: "Legality, Constitutionality, Jurisdiction",
#     7: "Crime and Punishment",
#     8: "Security and Defense",
#     9: "Health and Safety",
#     10: "Quality of Life",
#     11: "Economics",
#     12: "Capacity and Resources",
#     13: "Policy Description, Prescription, Evaluation",
#     14: "External Regulation and Reputation",
#     15: "Other"
# }

# x_axis_labels = {
#     1: ("Left", "Right"),
#     2: ("Disapproval", "Support"),
#     3: ("Assimilation", "Tradition"),
#     4: ("Secular Values", "Religious Values"),
#     5: ("Discrimination", "Equity"),
#     6: ("Unlawful", "Constitutional"),
#     7: ("Leniency", "Retribution"),
#     8: ("Vulnerable", "Protected"),
#     9: ("Harmful", "Safe"),
#     10: ("Discomfort", "Wellbeing"),
#     11: ("Cost", "Benefit"),
#     12: ("Insufficient", "Sufficient"),
#     13: ("Ineffective", "Effective"),
#     14: ("Criticized", "Respected"),
#     15: ("Undefined", "Undefined")
# }

# fig, ax = plt.subplots(figsize=(10, 8))

# for x, item in sdf.iterrows():
#     x = item['leaning']
#     y = item['frame']
#     label = y_labels[int(y)]
#     ax.scatter(x, y, color='blue')
#     # ax.text(x + 0.1, y, label, verticalalignment='center', fontsize=9)
#     ax.text(x + 0.1, y, f"{x:.2f}", va='center', fontsize=9)

# # Formatting
# ax.set_xlim(-5.5, 5.5)
# ax.set_xlabel("X Axis (frame-specific polarity)")

# # Add labels on top and bottom for each frame
# for y, (left, right) in x_axis_labels.items():
#     ax.text(-5.5, y, left, va='center', ha='right', fontsize=8, color='gray')
#     ax.text(5.5, y, right, va='center', ha='left', fontsize=8, color='gray')

# ax.set_ylim(0.5, 15.5)
# # ax.set_yticks(list(y_labels.keys()))
# # ax.set_yticklabels([y_labels[i] for i in y_labels])
# for y, label in y_labels.items():
#     ax.text(0, y, label, ha='center', va='center', fontsize=9, weight='bold')
#     ax.set_alpha(.4)

# ax.set_title('Framing Graph')

# plt.tight_layout()
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.show()



# # Turn off default y-axis labels and ticks
# ax.set_yticks([])
# ax.set_xlim(-6, 6)
# ax.set_ylim(0.5, 15.5)
# ax.set_xlabel("X Axis (frame-specific polarity)")

# # Draw manual y-axis labels in the center
# for y, label in y_labels.items():
#     ax.text(0, y, label, ha='center', va='center', fontsize=15, color='gray')

# # Add left and right pole labels for each frame
# for y, (left, right) in x_axis_labels.items():
#     ax.text(-5.99, y, left, va='center', ha='right', fontsize=8, color='gray')
#     ax.text(5.99, y, right, va='center', ha='left', fontsize=8, color='gray')

# # Aesthetics
# ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
# ax.set_title("Scatter Plot with Centered Frame Labels and X Value Annotations")
# ax.grid(True, axis='x', linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Frame labels for Y axis
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

# X axis label pairs for each frame
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

# Generate distinguishable colors for 15 frames
cmap = plt.cm.get_cmap("tab20", len(y_labels))
colors = {y: cmap(i) for i, y in enumerate(y_labels)}

fig, ax = plt.subplots(figsize=(12, 10))
# fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True)

# Draw background color bands and create legend handles
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

for x, item in sdf.iterrows():
    x = item['leaning']
    y = item['frame']
    label = y_labels[int(y)]
    ax.scatter(x, y, color='blue')
    # ax.text(x + 0.1, y, label, verticalalignment='center', fontsize=9)
    ax.text(x + 0.1, y, f"{x:.2f}", va='center', fontsize=9)

# # Axis formatting
# ax.set_xlim(-6, 6)
# ax.set_ylim(0.5, 15.5)
# ax.set_xlabel("X Axis (frame-specific polarity)")
# ax.set_yticks([])  # remove default y-ticks

# # Add left and right pole labels
# for y, (left, right) in x_axis_labels.items():
#     ax.text(-6, y, left, va='center', ha='right', fontsize=8, color='gray')
#     ax.text(6, y, right, va='center', ha='left', fontsize=8, color='gray')

# # Add vertical center line
# ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# # Add legend
# ax.legend(handles=patches, bbox_to_anchor=(1.02, 4), loc='upper left', borderaxespad=0.)

# # Title and grid
# ax.set_title("Frame Variation")
# ax.grid(True, axis='x', linestyle='--', alpha=0.3)
# plt.tight_layout()
# # fig.subplots_adjust(right=0.99)
# plt.savefig("framed_plot.png", dpi=300, bbox_inches='tight')

# plt.show()


# Axis formatting
ax.set_xlim(-6, 6)
ax.set_ylim(0.5, 15.5)
ax.set_xlabel("Leaning")
ax.set_ylabel("Frame")
ax.set_yticks([])  # remove default y-ticks

# Add left and right pole labels
for y, (left, right) in x_axis_labels.items():
    ax.text(-6, y, left, va='center', ha='right', fontsize=8, color='gray')
    ax.text(6, y, right, va='center', ha='left', fontsize=8, color='gray')

# Add vertical center line
ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

# Add legend
ax.legend(handles=patches, bbox_to_anchor=(1.09, 1), loc='upper left', borderaxespad=0.)

# Title and grid
ax.set_title("Framing Variation")
ax.grid(True, axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()