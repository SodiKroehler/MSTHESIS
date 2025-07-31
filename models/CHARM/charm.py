import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import re
from sentence_transformers import SentenceTransformer
import hdbscan

class CHARM:
    def __init__(self, n_topics=5, seed_topics=None, n_top_words=10, random_state=42):
        self.n_topics = n_topics
        self.seed_topics = seed_topics or []
        self.n_top_words = n_top_words
        self.random_state = random_state
        self.vectorizer = CountVectorizer(stop_words='english')

        self.MIN_DISTANCE_THRESHOLD = 0.5
        self.guideposts = { #temp, should be embeeddigns from seed_topics
            "gopher": [0.1, 0.2, 0.3, 0.4, 0.5],
            "star wars": [0.5, 0.4, 0.3, 0.2, 0.1]
        }

    #helers
    def gp_sims(self, para_embeddings):
        #para_embeddings should be a DataFrame with embeddings and guideposts_embeddings a dict

        for key in self.guideposts_embeddings.keys():
            if f"gp_{key}" not in para_embeddings.columns:
                para_embeddings[f"gp_{key}"] = np.nan
        
        for idx, row in para_embeddings.iterrows():
            pembeds = para_embeddings.at[idx, "embeddings"]#will be an n x e matrix
            gpembeds = np.array([self.guideposts[gp] for gp in self.guideposts.keys()])  # will be a k x e matrix
            sims = cosine_similarity([pembeds], gpembeds)
            #sims should now be a n x k matrix
            for gp_idx, gp in enumerate(self.guideposts.keys()):

                gpsum = np.sum(sims, axis=0) # is now a 1 x k matrix
                #at a gp_idx, this is the similarity between the para and the guidepost
                para_embeddings.at[idx, f"gp_{gp}"] = gpsum[gp_idx]
        
        return para_embeddings
    
    def pegger(self, embeddings):
        #should return a len(paras) x 1 array assigning each paragraph to a peg
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,      #? right? wrong ?
            metric='euclidean',      #eh? idek
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        return cluster_labels
    
    def coh(self, df):
        #already built on the mega server so, lets just use it there then
        return df


#     1. isolate each paragraph

# 2.    classify each paragraph based on similarity to guidepost
# 3. combine paragraphs into pegs
# 3. obtain coh scores for each ir, for each paragraph
# 4. this gives us three coords:
#     1. x: ir leaning
#     2. y: ir
#     3. z: peg
# 5. for each peg, if there does not exist an opposite peg, add ghost peg.
# 6. for each article, add sum score all. paragraphs, plus sum score ghost pegs, weighted by ir/leanign of chioce mapping.
    def main(self, df):
        #get a shared document id
        df["article_id"] = df.index + 1
        
        paras = pd.DataFrame(columns=["article_id", "text", "position"])
        for idx, row in df.iterrows():
            paras = row["text"].split('\n')
            idx = 0
            #remove any single line paragraphs (common in news articles)
            while idx < len(paras):
                num_sentences = len([sent for sent in re.split(r'[.!?]', paras[idx]) if sent.strip()])
                if num_sentences < 2:
                    paras[idx-1] += ' ' + paras[idx]
                    s = paras.pop(idx)
                else:
                    idx += 1
            for position, para in enumerate(paras):
                newpara  = {
                    "article_id": row["article_id"],
                    "text": para.strip(),
                    "position": position + 1
                }
                paras = paras.append(newpara, ignore_index=True)

        # step 1 complete
        # before Step 2, have to vectorize:
        #clean each paragraph
        paras["text"] = paras["text"].str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
        model = SentenceTransformer('all-MiniLM-L6-v2')

        paras["embeddings"] = paras["text"].apply(lambda x: np.random.rand(5))  # Mock embeddings for simplicity
        # paras["embeddings"] = paras["text"].apply(lambda x: model.encode(paras["text"].tolist())) 
        paras = self.gp_sims(paras)

        # step 2 now. need to combine like paragraphs into pegs.

        paras["peg"] = self.pegger(paras['embeddings'].tolist())
        #eventually this might be smarter, and take into accunt the guideposts, but for now, just use hdbscan.

        # step 3, get the coherence scores
        paras = self.coh(paras)

        # and we now have coords. we want to pivot, at least so we can index by x.
        # step 4, for each peg, if there does not exist an opposite peg, add ghost peg.
        #final desired result: a 3 x num_pegs x 7



        #  classify each paragraph based on similarity to guidepost
        #first thing w
