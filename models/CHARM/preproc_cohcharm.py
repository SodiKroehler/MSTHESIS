import pandas as pd

#final goal is to get
# A, P, peg, peg_x, peg_y, peg_z
#so each article will have len(A_p) * 7 (for irs) 
# some of those rows might have shared values of z, as they could be in the same peg
# #think we should do the ghost pegs thing at a later time, so we don't confuse everyone and idk can focus on getting this one first
# each should have a different value of x, since that depends on the current value of y


#currently, we have :
df = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/from_crc/cohcharm_longlong_jul30.csv', index_col=False)
# columns = ['Unnamed: 0', 'charm_article_id', 'text', 'position', 'IR', 'bucket',
#        'guidewords', 'likes', 'dislikes', 'attestation', 'input_text',
#        'logprob']
df.rename(columns={"charm_article_id": "A", "position": "P", "text": "p_text"}, inplace=True)
odf = df.copy()
#guidewords, likes, dislikes, attestation, and input_text are all "constants":
df = df.drop(columns=['Unnamed: 0', 'guidewords', 'likes', 'dislikes', 'attestation', 'input_text'])


#logprob is the log of the next tokens prob, so can't be > 0.
#lower logprob means lower prob, so less likely.
#lower average logprob means lower coherence (roughly)
#it's not standardized, so we'll standardize it here
df['logprobz'] = (df['logprob'] - df['logprob'].mean()) / df['logprob'].std() #now will be roughly -3 to 3

#pretty decent format so far, but we do have two rows for each ir, so need to combine those
def combine_buckets(group):

    plus = 0
    minus = 0
    for idx, row in group.iterrows():
        if row['bucket'] == '+':
            plus = group.at[idx, 'logprob']
        elif row['bucket'] == '-':
            minus = group.at[idx, 'logprob']

    #so logprobz is now between -3 and 3 and still lower values mean lower coherence
    # so if we have lower coherence with the minus bucket, we have some more coherence with the positive.
    #would be really suprising (dont think its possible) if the range would change here
    combined = plus - minus
    # z_score = round((combined - 0.5) / 0.1) #we don't need this - restandardize after the func
    group['coh'] = combined
    group['minus'] = minus
    group['plus'] = plus
    return group.head(1)

df = df.groupby(['A', 'P', 'IR'], group_keys=True).apply(combine_buckets, include_groups=False).reset_index()

# we want y to be numeric, so we'll map quick
ir_labels = {
    0: "Inspired",
    1: "Popular",
    2: "Moral",
    3: "Civic",
    4: "Economic",
    5: "Functional",
    6: "Ecological"
}
f_ir_labels = {v: k for k, v in ir_labels.items()}
df['IR_num'] = df['IR'].map(f_ir_labels)


# df = df[[
#     'A', 'P', 'p_text', 'IR', 'coh', 'IR_num'
# ]]

df.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/from_crc/cohcharm_base_file.csv', index=False)