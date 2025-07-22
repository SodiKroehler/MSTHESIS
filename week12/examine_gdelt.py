import pandas as pd

# Read both CSV files
df1 = pd.read_csv('~/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/combined_gdelt_ai.csv')
df2 = pd.read_csv('~/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/combined_gdelt_unfiltered2.csv')

# Extract 'source' from 'article_url' in both DataFrames
df1['source'] = df1['article_url'].str.extract(r'https?://([^/]+)/')
df2['source'] = df2['article_url'].str.extract(r'https?://([^/]+)/')

# Concatenate the DataFrames
dfc = pd.concat([df1, df2], ignore_index=True)

# Convert 'date' column to datetime
dfc['date'] = pd.to_datetime(dfc['date'], errors='coerce')

print("Min date:", dfc['date'].min())
print("Max date:", dfc['date'].max())