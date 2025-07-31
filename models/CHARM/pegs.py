import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
# from sentence_transformers import SentenceTransformer



# tent method: 
# 1. embed each paragraph.
# 2. for each paragraph, measure its distance to the guideposts
# 3. measure distance to all other paragraphs. 
# 4. for any distances less than threshold, group into peg.
# 5. either way lets make this a seperate funciton, so li and lin can review



# model = SentenceTransformer('all-MiniLM-L6-v2')

#STEP 1. GET EMBEDDINGS FOR EACH PARAGRAPH.
# we're splitting on para as that seems better than sentences - is that good?
#we won't use the model, so we can more easily understand the code.
para1 = "I am a gopher who enjoys dancing the macarena."
para2 = "Hans' ship completed the Kessel Run in twelve parsecs, a famous feat in Star Wars."
para3 = "As a gopher, I once watched Star Wars while dancing the macarena."
guideposts = ["gopher", "star wars"]

p1_embed = [0.75, 0.92, 0.90, 0.18, 0.77]
p2_embed = [0.12, 0.23, 0.35, 0.46, 0.57]
p3_embed = [0.99, 0.88, 0.77, 0.65, 0.54]
guideposts_embeddings = {
    "gopher": [0.1, 0.2, 0.3, 0.4, 0.5],
    "star wars": [0.5, 0.4, 0.3, 0.2, 0.1]
}

#STEP 2. MEASURE DISTANCE TO GUIDEPOSTS
# we probably should use euclidean since were mocking kmeans, but cosine is idk better? lin really needs to yell at me here.
# we want k scores for ecahc of the n paragraphs

para_embeddings = pd.DataFrame({
    "paraid": [1, 2, 3],
    "embeddings": [p1_embed, p2_embed, p3_embed]
})



def gp_sims(para_embeddings, guideposts_embeddings):

    for key in guideposts_embeddings.keys():
        if f"gp_{key}" not in para_embeddings.columns:
            para_embeddings[f"gp_{key}"] = np.nan
    
    for idx, row in para_embeddings.iterrows():
        pembeds = para_embeddings.at[idx, "embeddings"]#will be an n x e matrix
        gpembeds = np.array([guideposts_embeddings[gp] for gp in guideposts_embeddings.keys()])  # will be a k x e matrix
        sims = cosine_similarity([pembeds], gpembeds)
        #sims should now be a n x k matrix
        for gp_idx, gp in enumerate(guideposts_embeddings.keys()):

            gpsum = np.sum(sims, axis=0) # is now a 1 x k matrix
            #at a gp_idx, this is the similarity between the para and the guidepost
            para_embeddings.at[idx, f"gp_{gp}"] = gpsum[gp_idx]
    
    return para_embeddings

para_embeddings = gp_sims(para_embeddings, guideposts_embeddings)
#so this gives us a dataframe with the paraid, embeddings, and the similarity to each guidepost.
#to double check, compare para1 to gopher
print("Actual similarity of para1 to gopher:", cosine_similarity([p1_embed], [guideposts_embeddings["gopher"]]))
#and it should be equal to
print("Calculated similarity of para1 to gopher:", para_embeddings.at[0, "gp_gopher"])

# and we get: 
# Actual similarity of para1 to gopher: [[0.78834352]]
# Calculated similarity of para1 to gopher: 0.7883435235884447
#so this seems right


#STEP 3. MEASURE DISTANCE TO ALL OTHER PARAGRAPHS
#this one is simple. there might be a way to make it more efficient, but its computationally cheap and well probably redo this part anywayw
grand_old_opry = np.zeros((len(para_embeddings), len(para_embeddings)))
for i in range(len(para_embeddings)):
    for j in range(len(para_embeddings)):
        if i != j:
            grand_old_opry[i][j] = cosine_similarity([para_embeddings.at[i, "embeddings"]], [para_embeddings.at[j, "embeddings"]])[0][0]
            #yeah?
            #not sure how to validate, but yeah this seems good

# STEP 4. GROUP INTO PEGS
# now we need a threshold bit (tunable?) 
THRESHOLD = 0.5

                        
