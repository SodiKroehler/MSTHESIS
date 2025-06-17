import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


df = pd.read_csv('./procon_longer.csv')
df['text'] = df['point'].astype(str) + ' ' + df['explanation'].astype(str)
# df = df.sample(n=200, random_state=42)
df = df[df['title'] == 'Artificial Intelligence (AI)']



vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])


lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(X)
# Get topic distribution for each document
topic_distributions = lda.transform(X)
# Assign the most probable topic to each document
df['LDA'] = topic_distributions.argmax(axis=1)

topic_words = {}
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    top_features_ind = topic.argsort()[:-11:-1]
    top_features = [feature_names[i] for i in top_features_ind]
    # print(f"Topic {topic_idx}: {', '.join(top_features)}")
    topic_words[topic_idx] = ', '.join(top_features)

df['LDA_terms'] = df['LDA'].apply(lambda x: topic_words.get(x, ''))
df = df[['title', 'point', 'explanation', 'LDA', 'LDA_terms']]

df.to_csv('procon_lda2_ai.csv')

