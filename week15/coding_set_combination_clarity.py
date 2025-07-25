import pandas as pd
import re

#_________________________________CODING________________________________________

#renaming maps
poli_leaning_dict = {
    "FAR_LEFT": -3,
    "LEFT": -2,
    "SLIGHT_LEFT": -1,
    "CENTER": 0,
    "SLIGHT_RIGHT": 1,
    "RIGHT": 2,
    "FAR_RIGHT": 3,
    "NONE": 0,
    "left": -2,
    "right": 2,
    "center": 0,
    "none": 0,
    "global": None,
    '': None
}
#sources leanings from allsides on jul16:
source_leaning_dict = {
    "nytimes.com": "LEAN_LEFT", #lean_left
    "msnbc.com": "LEFT",
    "theguardian.com": "LEFT",
    "reuters.com": "CENTER",
    "apnews.com": "LEFT",
    "npr.org": "LEAN_LEFT",
    "foxnews.com": "RIGHT",
    "dailycaller.com": "RIGHT",
    "washingtonexaminer.com": "LEAN_RIGHT",
    "bbc.com": "CENTER",
    "bbc.co.uk": "CENTER",
    "aljazeera.com": "LEAN_LEFT",
    "english.news.cn": "LEFT",
}
small_poli_leaning_dict = {
    "FAR_LEFT": -1,
    "LEFT": -1,
    "SLIGHT_LEFT": -1,
    "CENTER": 0,
    "SLIGHT_RIGHT": 1,
    "RIGHT": 1,
    "FAR_RIGHT": 1,
}
want_to_see_dict = {
    "Yes, this seems like the content I usually read": 2,
    "Yes, but this seems to be outside my usual content": 1,
    "No, this is outside my usual content, and I don't like it": -2,
    "No, this seems like the content I usually read, but I don't like this piece": -1,
    "This question is irrelevant here": 0,
    "THE_SAME": 2,
    "SIMILARLY": 1,
    "DIFFERENTLY": -1,
    "WOULD_NOT_HAVE_WRITTEN_ABOUT_IT": 0,
    "VERY_DIFFERENTLY": -2
}

ai_leaning_dict = {
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "OPTIMISTIC": 1,
    "PESSIMISTIC": -1,
    "PRO_AI": 1,
    "ANTI_AI": -1
}

ai_reg_leaning_dict = {
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "PRO_REGULATION": 1,
    "ANTI_REGULATION": -1
}

imm_leaning_dict = {
    "UNDEFINED": None,
    "NEUTRAL": 0,
    "PERMISSIVE": 1,
    "STRICT": -1,
    "PRO_IMMIGRATION": 1,
    "ANTI_IMMIGRATION": -1
}

bias_amount_dict = {
    "LOW": 1,
    "NONE": 0,
    "MEDIUM": 2,
    "HIGH": 3,
    "UNDEFINED": None
}

rater_dict = {
    "sodi.kroehler@gmail.com": "sodi",
    "emmetmathieu@gmail.com": "emmet",
    "sodikroehler@gmail.com": "sodi",
    "emmet.mathieu@gmail.com": "emmet"
}

def standardize_rating_df(df):
    required_columns = [
        'user_poli_leaning', 'user_agree_with', 'user_wants_to_see', 'user_article_subject',
        'user_ai_leaning', 'user_aireg_leaning', 'user_imm_leaning', 'user_bias_amnt', 'user_bias_cause',
        'gpt_leaning', "set_number", 'source_url',
        'source', 'rater_email', 'coding_date', 'coding_time', 'gkg_id', 'title', 'text', 'date'
    ]
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"DataFrame must contain the following columns: {missing_cols}")
    
    #clean all columns
    gpt_leaning_name = f"{df.loc[0, 'set_number']}_gpt_leaning"
    df['user_poli_leaning_num'] = df["user_poli_leaning"].map(poli_leaning_dict)
    df['user_agree_with'] = df['user_agree_with'] #just t/f
    df['user_article_subject'] = df['user_article_subject'] #missing ai reg here, but otherwise the same
    df['user_wants_to_see_num'] = df['user_wants_to_see'].map(want_to_see_dict)
    df['user_ai_leaning_num'] = df['user_ai_leaning'].map(ai_leaning_dict)
    df['user_aireg_leaning_num'] = df['user_aireg_leaning'].map(ai_reg_leaning_dict)
    df['user_imm_leaning_num'] = df['user_imm_leaning'].map(imm_leaning_dict)
    df['user_bias_amnt_num'] = df['user_bias_amnt'].map(bias_amount_dict)
    df['user_bias_cause'] = df['user_bias_cause'].fillna('')  #also just leaving the same here too
    df[gpt_leaning_name] = df['gpt_leaning'].map(poli_leaning_dict)
    df['source_leaning'] = df['source'].map(source_leaning_dict)
    df['source_leaning'] = df['source_leaning'].map(poli_leaning_dict)
    df['rater'] = df['rater_email'].map(rater_dict)


    df['set_coded_in'] = df['set_number']

    #manually pivoting wider, but sometimes helps avoid confusion.
    for rater in ["sodi", "emmet"]:
        for row in df[df['rater'] == rater].itertuples():
            df.loc[df['source_url'] == row.source_url, f'user_poli_leaning_num_{rater}'] = str(row.user_poli_leaning_num)
            df.loc[df['source_url'] == row.source_url, f'user_agree_with_{rater}'] = str(row.user_agree_with)
            df.loc[df['source_url'] == row.source_url, f'user_article_subject_{rater}'] = str(row.user_article_subject)
            df.loc[df['source_url'] == row.source_url, f'user_wants_to_see_num_{rater}'] = str(row.user_wants_to_see_num)
            df.loc[df['source_url'] == row.source_url, f'user_ai_leaning_num_{rater}'] = str(row.user_ai_leaning_num)
            df.loc[df['source_url'] == row.source_url, f'user_aireg_leaning_num_{rater}'] = str(row.user_aireg_leaning_num)
            df.loc[df['source_url'] == row.source_url, f'user_imm_leaning_num_{rater}'] = str(row.user_imm_leaning_num)
            df.loc[df['source_url'] == row.source_url, f'user_bias_amnt_num_{rater}'] = str(row.user_bias_amnt_num)
            df.loc[df['source_url'] == row.source_url, f'user_bias_cause_{rater}'] = str(row.user_bias_cause)

    columns_to_check_for_coding = ['coding_date', 'coding_time', 'user_poli_leaning']
    columns_to_check_for_coding = [ col for col in columns_to_check_for_coding if col in df.columns ]
    df['coded'] = df[columns_to_check_for_coding].notna().any(axis=1)
    df = df[df['coded'] == True]  # Only keep coded articles

    columns_to_exclude = [col for col in df.columns if col.startswith('user_') and not re.match(r'^user_.*(_emmet|_sodi)$', col)]
    df = df.drop(columns=columns_to_exclude, errors='ignore')

    #then we drop duplicates, as well as the copies from the long version.
    df = df.drop_duplicates(subset=['source_url'], keep='last').dropna(subset=['source_url'])

    
    #set a custom id for each article/set combination - there'll be no duplicates by this point
    df['article_set_unique_id'] = [id for id in range(1, len(df) + 1)]
    #convention has been to leave 1k between sets, just to be safe and for ease of use
    df['article_set_unique_id'] = lambda x: x['article_set_unique_id'] + (x['set_number'].str[-1].astype(int) * 1000)
    #we also make it a string, sometimes that helps avoid confusion
    df['article_set_unique_id'] = df['set_number'].astype(str) + '_' + df['article_set_unique_id'].astype(str)

    return df
    
#Set 1 - coded
df_jul8 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/MSTHESIS_BACKEND_SHEET-completedJuly8.csv')
df_jul8.columns = ['index', 'title', 'text', 'date', 'source_url', 'pull', 'source',
       'source_leaning', 'clean', 'gpt_leaning', 'LLM_RESPONSE', 'set_number',
       'rater_email', 'user_poli_leaning', 'user_agree_with', 'user_article_subject',
       'user_wants_to_see', 'coding_date', 'coding_time', 'ID']
# #orig: ['index', 'title', 'text', 'date', 'article_url', 'pull', 'source',
#        'source_leaning', 'clean', 'gpt_leaning', 'LLM_RESPONSE', 'set_number',
#        'rater_email', 'human_leaning', 'human_retweet', 'human_subject',
#        'human_agree_with', 'coding_date', 'coding_time', 'ID']

for col in ['user_ai_leaning', 'user_aireg_leaning', 'user_imm_leaning', 'user_bias_amnt', 'user_bias_cause', 'gkg_id']:
    df_jul8[col] = None

#dropping early as weird things are happening
df_jul8 = df_jul8.dropna(subset=['source_url', 'user_poli_leaning'])

df_jul8 = standardize_rating_df(df_jul8)



#Set 2 -coded
df_jul16 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week14/MSTHESIS_BACKEND_SHEET - PRODUCTION_jul16.csv')
df_jul16.rename(columns={
    'article_url': 'source_url',
}, inplace=True)
df_jul16['set_number'] = 'set2'
#dropping early as weird things are happening
df_jul16 = df_jul16.dropna(subset=['source_url', 'user_poli_leaning'])
df_jul16 = standardize_rating_df(df_jul16)




#Set 3 - coded
df_jul24 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/MSTHESIS_BACKEND_SHEET - PRODUCTION - pulled jul24.csv')
df_jul24.rename(columns={
    'article_url': 'source_url',
}, inplace=True)
#dropping early as weird things are happening
df_jul24 = df_jul24.dropna(subset=['source_url', 'user_poli_leaning'])
df_jul24 = standardize_rating_df(df_jul24)



#Set 4
#current state is coding
df_jul25 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/set4_sampled_rows.csv')
for col in ['user_poli_leaning', 'user_agree_with', 'user_wants_to_see', 
            'user_article_subject', 'user_ai_leaning', 'user_aireg_leaning', 
            'user_imm_leaning', 'user_bias_amnt', 'user_bias_cause', 'gpt_leaning', 'coding_date', 'coding_time']:
    if col not in df_jul25.columns:
        df_jul25[col] = None

df_jul25['set_number'] = 'set4'
#hack bc im dumb:
df_jul25['coding_date'] = "not yet done"
df_jul25 = standardize_rating_df(df_jul25)




combined_wide_form = pd.concat([df_jul8, df_jul16, df_jul24, df_jul25], ignore_index=True)
combined_wide_form = combined_wide_form.drop_duplicates(subset=['source_url'], keep='last').dropna(subset=['source_url'])
def clean_for_humans(text):
    if pd.isna(text):
        return ''
    for phrase in ['NPR', 'AP', 'BBC', 'NYT']:
        text = re.sub(rf'(?i)(?<=\W){re.escape(phrase)}(?=\W)', '[SOURCE]', text)
    for phrase in ['Reuters', 'Associated Press', 'The Guardian', 
                     'The New York Times', 'MSNBC', 'Fox News', 'Daily Caller', 'Washington Examiner', 
                     'Consider This', 'Short Wave', 'The Daily', 'News Hour', 'Al Jazeera', 'Xinhua',
                     'licensing@dailycallernewsfoundation.org', 'The Times ']:
        text = re.sub(phrase, '[SOURCE]', flags=re.IGNORECASE, string=text)
    return text.strip()
combined_wide_form['human_clean_text'] = combined_wide_form['text'].apply(clean_for_humans)
combined_wide_form['human_clean_title'] = combined_wide_form['title'].apply(clean_for_humans)


combined_wide_form.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_wide_form_pre_jul25.csv', index=False)


#pivot wider
sodi_cols = [c for c in combined_wide_form.columns if c.endswith('_sodi')]
emmet_cols = [c for c in combined_wide_form.columns if c.endswith('_emmet')]
meta_cols = [c for c in combined_wide_form.columns if not (c.endswith('_sodi') or c.endswith('_emmet'))]


sodi_renamed = {col: col.replace('_sodi', '') for col in sodi_cols}
emmet_renamed = {col: col.replace('_emmet', '') for col in emmet_cols}


sodi_df = combined_wide_form[meta_cols + sodi_cols].rename(columns=sodi_renamed)
sodi_df['rater'] = 'sodi'

emmet_df = combined_wide_form[meta_cols + emmet_cols].rename(columns=emmet_renamed)
emmet_df['rater'] = 'emmet'
long_df = pd.concat([sodi_df, emmet_df], ignore_index=True)

long_df['set4_id'] = range(1, len(long_df) + 1)
long_df['set4_id'] = long_df['set4_id'] + (long_df['set_number'].str[-1].astype(int) * 1000)
long_df['set3_ID'] = long_df['set3_ID'].fillna(long_df['set4_id'])
long_df.drop(columns=['set4_id'], inplace=True)
breakpoint()

# #reverse the num labels
numcols = ['user_poli_leaning_num', 'user_wants_to_see_num', 'user_ai_leaning_num', 
            'user_aireg_leaning_num', 'user_imm_leaning_num', 'user_bias_amnt_num']
# Flipped (reversed) versions of the mapping dicts
flipped_poli_leaning_dict = {v: k for k, v in poli_leaning_dict.items() if v is not None}
flipped_want_to_see_dict = {v: k for k, v in want_to_see_dict.items() if v is not None}
flipped_ai_leaning_dict = {v: k for k, v in ai_leaning_dict.items() if v is not None}
flipped_ai_reg_leaning_dict = {v: k for k, v in ai_reg_leaning_dict.items() if v is not None}
flipped_imm_leaning_dict = {v: k for k, v in imm_leaning_dict.items() if v is not None}
flipped_bias_amount_dict = {v: k for k, v in bias_amount_dict.items() if v is not None}

flipped_maps = [
    flipped_poli_leaning_dict,
    flipped_want_to_see_dict,
    flipped_ai_leaning_dict,
    flipped_ai_reg_leaning_dict,
    flipped_imm_leaning_dict,
    flipped_bias_amount_dict
]
for idx, col in enumerate(numcols):
    if col in long_df.columns:
        new_col_name = col.replace('_num', '')
        long_df[new_col_name] = long_df[col].astype(float).map(flipped_maps[idx])
        long_df.drop(columns=[col], inplace=True)

long_df.rename(columns={"source_url": "article_url"}, inplace=True)

long_df = long_df[[
    "gkg_id", "timestamp", "themes", "DF_IDX", "url", "url_tuple", "URL_new",
    "images", "authors", "movies", "tags", "clean", "gpt_leaning",
    "LLM_RESPONSE", "set_number", "rater_email", "coding_date", "coding_time",
    "old_bad_id", "article_url", "pull", "source", "title", "text", "date",
    "human_clean_text", "human_clean_title", "ID", "user_poli_leaning",
    "user_agree_with", "user_wants_to_see", "user_article_subject",
    "user_ai_leaning", "user_aireg_leaning", "user_imm_leaning",
    "user_bias_amnt", "user_bias_cause", "set3_ID"
]]
long_df.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_long_form_pre_jul25.csv', index=False)