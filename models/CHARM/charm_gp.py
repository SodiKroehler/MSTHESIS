def gp_sims(self, para_embeddings, guideposts_embeddings=None):
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


    