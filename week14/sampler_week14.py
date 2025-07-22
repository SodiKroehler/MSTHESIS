import pandas as pd
import re

odf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/final_july_combined_immandai.csv')
odf = odf.dropna(subset=['human_clean_text'])

sample_imm = odf[odf['pull'] == 'july_all25_imm'].sample(n=100, random_state=42)
sample_ai = odf[odf['pull'] == 'july_all25_ai'].sample(n=100, random_state=42)
other_sample = odf[(odf['pull'] != 'july_all25_imm') & (odf['pull'] != 'july_all25_ai')].sample(n=100, random_state=42)

sdf = pd.concat([sample_imm, sample_ai, other_sample], ignore_index=True)
sdf['set_number'] = 'set3'
sdf.drop(columns=['human_leaning', 'human_retweet', 'human_subject', 'human_agree_with', 'source_leaning'], inplace=True)

#redoing this because it's not very good rn:

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



sdf['human_clean_text'] = sdf['text'].apply(clean_for_humans)
sdf['human_clean_title'] = sdf['title'].apply(clean_for_humans)

sdf['user_poli_leaning'] = ""
sdf['user_agree_with'] = ""
sdf['user_wants_to_see'] = ""
sdf['user_article_subject'] = ""
sdf['user_ai_leaning'] = ""
sdf['user_aireg_leaning'] = ""
sdf['user_imm_leaning'] = ""
sdf['user_bias_amnt'] = ""
sdf['user_bias_cause'] = ""

sdf['rater_email'] = 'sodikroehler@gmail.com'

sdf_em = sdf.copy()
sdf_em['rater_email'] = 'emmet_mathieu@gmail.com'
sdf = pd.concat([sdf, sdf_em], ignore_index=True)

sdf['set3_ID'] = range(1, len(sdf) + 1)
sdf['set3_ID'] = sdf['set3_ID'] + 3000

sdf.to_csv('set3_sampled_forUsers.csv', index=False)



#ai_df

# adf = sdf[sdf['rater_email'] == 'emmet_mathieu@gmail.com'].copy()
# adf.drop(columns=['user_poli_leaning', 'user_agree_with', 'user_wants_to_see', 'user_article_subject',
#                  'user_ai_leaning', 'user_imm_leaning', 'user_bias_amnt', 'user_bias_cause', 'rater_email', 'set3_ID', 'user_aireg_leaning'], inplace=True)

# adf['gpt_poli_leaning'] = ""
# adf['gpt_article_subject'] = ""
# adf['gpt_ai_leaning'] = ""
# adf['gpt_aireg_leaning'] = ""
# adf['gpt_imm_leaning'] = ""
# adf['gpt_bias_amnt'] = ""
# adf['gpt_bias_cause'] = ""
# adf['LLM_RESPONSE'] = ""
# adf['gpt_justification'] = ""

# adf['boyd_poli_leaning'] = ""

# adf.to_csv('set3_sampled_forAI.csv', index=False)

