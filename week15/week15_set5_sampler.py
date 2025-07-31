import pandas as pd


cllmdf = pd.read_csv('from_gpts/set4_withclaude_full.csv', index_col=False)
wdf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_wide_form_pre_jul25.csv', index_col=False)
gpt4odf = pd.read_csv('from_gpts/set4_withgpt4o_full.csv', index_col=False)




gpt_urls = gpt4odf['article_url'].unique()
cll_urls = cllmdf['article_url'].unique()
wdf_urls = wdf['source_url'].unique()

# wdf_not_in_gpt = [url for url in wdf_urls if url not in gpt_urls]
# gpt_not_in_wdf = [url for url in gpt_urls if url not in wdf_urls]

#there was issues with set4, so we made a new set5
#thus, there was a mismatch with what was llm coded

#we're going to get the missing rows, and rerun them, and then we'll add them in later and remove the extras

# gpt_extra_urls = [url for url in wdf_urls if url not in gpt_urls]
# #both dfs were the same, so we'll just save one for the extras
# llm_extras = wdf[wdf['source_url'].isin(gpt_extra_urls)].copy()
# llm_extras.to_csv('set5_llm_extras.csv', index=False)

#now we got them:
gpt_extras = pd.read_csv('from_gpts/set5extras_withgpt4o_full.csv', index_col=False)
claude_extras = pd.read_csv('from_gpts/set5extras_withclaude_full.csv', index_col=False)

#can combine them:
gpt = pd.concat([gpt4odf, gpt_extras], ignore_index=True)
claude = pd.concat([cllmdf, claude_extras], ignore_index=True)

gpt.to_csv('set5_withgpt4o_full.csv', index=False)
claude.to_csv('set5_withclaude_full.csv', index=False)

#and finally, write the final long form for coding:
#we know wide and long are alreayd okay, and we didn't change anything, so we'll just load from long
ldf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_long_form_jul30.csv', index_col=False)
ldf = ldf.merge(gpt[['article_url', 'gpt_summary']], on='article_url', how='left')


#neither of these have isCoded, and that's important so:
jul_30 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/MSTHESIS_BACKEND_SHEET - PRODUCTION - pulledjul30 - midR4.csv')
sodiCoded = jul_30[jul_30['rater_email'] == 'sodikroehler@gmail.com'][['article_url', 'isCoded']]
emmetCoded = jul_30[jul_30['rater_email'] == 'emmetb@umich.edu'][['article_url', 'isCoded']]
for idx, row in ldf.iterrows():
    if row['rater_email'] == 'sodi.kroehler@gmail.com':
        if row['article_url'] in sodiCoded['article_url'].values:
            ldf.at[idx, 'isCoded'] = sodiCoded[sodiCoded['article_url'] == row['article_url']]['isCoded'].values[0]
        else:
            ldf.at[idx, 'isCoded'] = None
    elif row['rater_email'] == 'emmetmathieu@gmail.com':
        if row['article_url'] in emmetCoded['article_url'].values:
            ldf.at[idx, 'isCoded'] = emmetCoded[emmetCoded['article_url'] == row['article_url']]['isCoded'].values[0]
        else:
            ldf.at[idx, 'isCoded'] = None
ldf['rater_email'] = ldf['rater_email'].replace({
    'sodi.kroehler@gmail.com': 'sodikroehler@gmail.com',
    'emmetmathieu@gmail.com': 'emmet.mathieu@gmail.com'
})
ldf['isCoded'].fillna(False, inplace=True)



coded_true_urls = jul_30[jul_30['isCoded'] == True]['article_url'].unique()
ldf_coded_true_urls = ldf[ldf['isCoded'] == True]['article_url'].unique()
ldf_all_urls = ldf['article_url'].unique()

missing_or_mismatched = []
for url in coded_true_urls:
    if url in ldf_all_urls:
        if url not in ldf_coded_true_urls:
            missing_or_mismatched.append(url)
    # If not in ldf, that's fine per your instructions

if missing_or_mismatched:
    print("These article_urls are in jul_30 and coded True, but in ldf and not coded True:", missing_or_mismatched)
else:
    print("All True-coded article_urls in jul_30 are either not in ldf or coded True in ldf.")

ldf.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/set5_sampled_rows.csv', index=False)
