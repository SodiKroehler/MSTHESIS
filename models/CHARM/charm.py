import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import re
# from sentence_transformers import SentenceTransformer, util
import hdbscan
from sklearn.cluster import KMeans

class CHARM:
    def __init__(self, ir_custom_scores, ir_labels, stances):
        self.vectorizer = CountVectorizer(stop_words='english')

        self.MIN_DISTANCE_THRESHOLD = 0.5
        self.guideposts = { #temp, should be embeeddigns from seed_topics
            "gopher": [0.1, 0.2, 0.3, 0.4, 0.5],
            "star wars": [0.5, 0.4, 0.3, 0.2, 0.1]
        }
        self.ir_custom_scores = ir_custom_scores #should have all stances as keys
        self.ir_labels = ir_labels
        self.f_ir_labels = {v: k for k, v in self.ir_labels.items()}
        self.stances = stances

    #helpers

    
    def sims(self, para_embeddings):
        #takes in df of embeddings of length P
        # expected to be a single column, and will be parsed out and padded to the longest paragraph length, hereafter called N, 
        # each embedding is expected to be from the same model, with size called now e
        # returns a e x e matrix of similarities
        if not isinstance(para_embeddings, pd.DataFrame):
            raise ValueError("para_embeddings must be a pandas DataFrame")
        N = para_embeddings['embeddings'].apply(lambda x: len(x)).max() + 1
        P = para_embeddings.shape[0]

        sim_matrix = np.zeros((P, P))
        expanded_para_embeddings = pd.DataFrame

        for i, irow in para_embeddings.iterrows():
            for j, jrow in para_embeddings.iterrows():
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif sim_matrix[i, j] > 0:
                    continue #already found
                else:
                    iembeds = para_embeddings.at[j, "embeddings"]#will be an <n x e matrix
                    jembeds = para_embeddings.at[i, "embeddings"]
                    #need to pad it for reliability
                    for embeds in [iembeds, jembeds]:
                        if len(embeds) < N:
                            embeds = np.append(embeds, np.zeros((N - len(embeds), len(embeds[0]))), axis=0)

                    sims = util.cos_sim(iembeds, jembeds)
                    #sims should be a num from -1 to 1
                    sim_matrix[i, j] = sims

        return sim_matrix


    
    def kmeans_pegger(self, embeddings, k=None):
        #should take in a list of embeddings and return a list of the same lengths with the cluster labels for each row
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels
    
    def hdbscan_pegger(self, embeddings):
        # embeddings is a pandas column filled with a len(para) x e matrix
        if not (isinstance(embeddings.iloc[0], np.ndarray) and embeddings.iloc[0].ndim == 2):
            raise ValueError("embeddings must be a pandas Series of 2D numpy arrays")
        para_embeddings = np.stack([e.mean(axis=0) for e in embeddings])
        #should return a len(paras) x 1 array assigning each paragraph to a peg
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=4,      #? right? wrong ?
            metric='euclidean',      #eh? idek
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(para_embeddings)
        return cluster_labels
    
    def popmat(self, df):
        k = df['peg'].nunique()
        mat = np.empty((11, k, 7), dtype=list)  # 11 for x, y, z; k for pegs; 7 for irs
        #really x should be -3 to 3, but its not enforced so its safer to have 5 on each side

        for idx, row in df.iterrows():
            peg = row['peg']
            ir_num = row['IR_num']
            coh = row['coh']

            #for now, we're doing this w an array. exact values might be better for the frame quality bit?
            x = int(coh + 6.0) # 0 should be 6
            negx = int(-coh + 6.0) #for easier ghost pegs

            #init stuff
            mat[x, peg, ir_num] = [] if mat[x, peg, ir_num] is None else mat[x, peg, ir_num]
            mat[negx, peg, ir_num] = [] if mat[negx, peg, ir_num] is None else mat[negx, peg, ir_num]

            # Map the values to the matrix
            mat[x, peg, ir_num].append(f"{row['A']}_{row['P']}")  # A_P as a string

            #ghost pegs
            if len(mat[negx, peg, ir_num]) < len(mat[x, peg, ir_num]): #inited already so cant be none
                mat[negx, peg, ir_num].append(f"ghost_{row['A']}_{row['P']}")  # A_P as a string
        return mat
    
    def get_num_score(self, stance, ir_num, x):

        ir_label = self.ir_labels[ir_num]
        ir_score = self.ir_custom_scores.get(ir_label).get("+").get(stance)
        if x < 6:
            ir_score = self.ir_custom_scores.get(ir_num).get("-").get(stance)
        
        return ir_score * x

    def make_mat(self, df, edf):
        # should take in a pd dataframe with 'A', 'P', 'p_text', 'IR', 'coh', 'IR_num'
        #does:
        # 1. isolate each paragraph
        # 2.    classify each paragraph based on similarity to guidepost
        # 3. combine paragraphs into pegs
        # 3. obtain coh scores for each ir, for each paragraph
        # 4. this gives us three coords:
        #     1. x: ir leaning
        #     2. y: ir
        #     3. z: peg
        # 5. for each peg, if there does not exist an opposite peg, add ghost peg.
        # 6. for each article, add sum score all. paragraphs, plus sum score ghost pegs, weighted by ir/leanign of chioce mapping.




        #removing preprocssing since we did it separately. 
        #also can be we just need the right structure for here

        # step 1 complete
        # before Step 2, have to vectorize:
        #clean each paragraph
        df["p_text"] = df["p_text"].str.replace(r'\s+', ' ', regex=True).str.strip().str.lower()
        # model = SentenceTransformer('all-MiniLM-L6-v2')

        # # df["embeddings"] = df["p_text"].apply(lambda x: np.random.rand(5))  # Mock embeddings for simplicity
        # df["embeddings"] = df["p_text"].apply(lambda x: model.encode(df["p_text"].tolist()))
        df["embeddings"] = df["p_text"].apply(lambda x: edf[edf['p_text'] == x]['embeddings'].values[0] if not edf[edf['p_text'] == x].empty else np.random.rand(5))  # Mock embeddings for simplicity
 


        # paras = self.gp_sims(paras) #later customization here - probably should be an override method
        # sim_matrix = self.sims(paras)

        # step 2 now. need to combine like paragraphs into pegs.
        k = len(df)
        #too many though. don't want to do elbow as we only want a max
        df["peg"] = self.hdbscan_pegger(df['embeddings'])
        #eventually this might be smarter, and take into accunt the guideposts with kmeans, but for now, just use hdbscan.

        # step 3 is already done before


        # and we now have coords. we want to pivot, at least so we can index by x.
        # step 4, for each peg, if there does not exist an opposite peg, add ghost peg.
        
        #final desired result: a 11 x num_pegs x 7
        mat = self.popmat(df)

        #since were doing things simple for now
        df['x'] = df['coh'] + 6.0  # 0 should be 6
        df['y'] = df['IR_num']
        df['z'] = df['peg']

        

        for stance in self.stances:
            df[f'charm_{stance}_num_for_p'] = df.apply(
                lambda row: self.get_num_score(stance, row['IR_num'], row['x']), axis=1
            )

        def custom_agg_for_A(df):
            if df.empty:
                raise ValueError("DataFrame is empty")
            for stance in self.stances:
                df['charm_' + stance + '_num_for_A'] = df['charm_' + stance + '_num_for_p'].mean()
            return df.head(1)
        
        fdf = df.groupby(['A'], group_keys=True).apply(
            custom_agg_for_A
        ).reset_index(drop=True)

        fdf = fdf[['A', 'charm_poli_num_for_A', 'charm_ai_num_for_A',
                   'charm_aireg_num_for_A', 'charm_imm_num_for_A']]
        return fdf, mat

if __name__ == "__main__":
    print('ehllo child')