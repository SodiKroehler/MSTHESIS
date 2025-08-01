import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import official_maps as maps
from urllib.parse import urlparse


sdf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_wide_form_pre_jul25.csv', index_col=False)
cmbdf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/complete_all_articles_jul25.csv', index_col=False)
df = cmbdf.copy()

# num_articles_in_cmbdf = sdf[sdf['source_url'].isin(cmbdf['source_url'])].shape[0]

df['source'] = df['source_url'].str.lower().str.split('/').str[2].str.replace('www.', '', regex=False)
df['source'] = df['source'].replace('bbc.co.uk', 'bbc.com')
df['source'] = df['source'].replace('foxbusiness.com', 'foxnews.com')
df['source_leaning'] = df['source'].map(maps.source_leaning_map)


ordering = maps.source_leaning_map.keys()
ordering = [s for s in ordering if s not in ['foxbusiness.com', 'bbc.co.uk']]



plt.figure(figsize=(16, 6))
ax = sns.countplot(data=df, x='source', hue='source_leaning', dodge=False, palette=maps.hue_colors, order=ordering)
plt.xlabel('Source')
plt.ylabel('Number of Articles')
plt.title('Number of Articles per Source (Colored by Leaning)')
plt.legend(title='Leaning')
plt.xticks(rotation=45, ha='right')

# Add counts on top of bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black', rotation=0)

plt.tight_layout()
plt.show()
