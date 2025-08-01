import pandas as pd
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import cohen_kappa_score
import official_maps as maps

#so cmbdf should have all possible rows, but only limited things about them.
cmbdf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/complete_all_articles_jul25.csv', index_col=False)
#wdf should have everything that we've coded, and everything we know of it
wdf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_wide_form_pre_jul25.csv', index_col=False)
# ['user_poli_leaning_num_sodi', 'user_poli_leaning_num_emmet',
#        'user_agree_with_sodi', 'user_agree_with_emmet',
#        'user_article_subject_sodi', 'user_article_subject_emmet',
#        'user_wants_to_see_num_sodi', 'user_wants_to_see_num_emmet',
#        'user_ai_leaning_num_sodi', 'user_ai_leaning_num_emmet',
#        'user_aireg_leaning_num_sodi', 'user_aireg_leaning_num_emmet',
#        'user_imm_leaning_num_sodi', 'user_imm_leaning_num_emmet',
#        'user_bias_amnt_num_sodi', 'user_bias_amnt_num_emmet',
#        'user_bias_cause_sodi', 'user_bias_cause_emmet', 'set_1_gpt_leaning',
#        'set_2_gpt_leaning', 'set_3_gpt_leaning', 'set_4_gpt_leaning',
#        'source_leaning', 'coding_date_sodi', 'coding_date_emmet',
#        'coding_time_sodi', 'coding_time_emmet', 'pull', 'source', 'title',
#        'text', 'date', 'gkg_id', 'set_number', 'source_url', 'coded',
#        'gpt_leaning', 'user_agree_with_nan', 'user_article_subject_nan',
#        'user_bias_cause_nan', 'coding_date_nan', 'coding_time_nan',
#        'article_set_unique_id', 'human_clean_text', 'human_clean_title']

#seems to be missing allsides, baly, spinde. also we know it's missing set4_gpt, even though it has that column
#this maybe should get put into the wide form, but for now we'll just do it here
#since its not in cmbdf, we ahve to get from orignal files:
asdf1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/ALLSIDES/ALLSIDES/final_jul25allsides.csv')
# ['source_url', 'pull', 'title', 'text', 'date', 'gkg_id','allside_bias']
asdf2 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/ALLSIDES/manual_pull_jul29/final_manual_scraper_allsides.csv')
# ['source_url', 'title', 'text', 'date', 'pull', 'full_text','allsides_leaning']

allsides_dict = {}
for idx, row in asdf1.iterrows():
    if row['source_url'] not in allsides_dict:
        allsides_dict[row['source_url']] = row['allside_bias']
for idx, row in asdf2.iterrows():
    if row['source_url'] not in allsides_dict:
        allsides_dict[row['source_url']] = row['allsides_leaning']


wdf['allsides_leaning'] = wdf['source_url'].map(allsides_dict)

#_____________LLMSSS_____________#
# we do actually have a lot of the chatgpt ones, so that might be cool to do

# CLAUDEEE
cllmdf = pd.read_csv('set4_withclaude_full.csv', index_col=False)
cllmdf = cllmdf.rename(columns={col: f'claude_{col}' for col in cllmdf.columns})
cllmdf = cllmdf.rename(columns={'claude_article_url': 'source_url'})
cllmdf = cllmdf.drop(columns=['claude_LLM_RESPONSE'])

# ['source_url', 'claude_gpt_summary', 'claude_gpt_poli_leaning',
#        'claude_gpt_poli_justification', 'claude_gpt_article_subject',
#        'claude_gpt_ai_leaning', 'claude_gpt_ai_justification',
#        'claude_gpt_aireg_leaning', 'claude_gpt_aireg_justification',
#        'claude_gpt_imm_leaning', 'claude_gpt_imm_justification',
#        'claude_gpt_primary_bias_amnt', 'claude_gpt_primary_bias_cause',
#        'claude_LLM_RESPONSE']

wdf = wdf.merge(cllmdf, on='source_url', how='left')


# GPT4o
gpt4odf = pd.read_csv('set4_withgpt4o_full.csv', index_col=False)
gpt4odf = gpt4odf.rename(columns={col: f'gpt4o_{col}' for col in gpt4odf.columns})
gpt4odf = gpt4odf.rename(columns={'gpt4o_article_url': 'source_url'})
gpt4odf = gpt4odf.drop(columns=['gpt4o_LLM_RESPONSE'])

wdf = wdf.merge(gpt4odf, on='source_url', how='left')


#baly
# balydf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/BALY/baly_jul25.csv', index_col=False)
wdf['baly'] = None


#spinde
# spindedf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/SPINDE/spinde_jul25.csv',
wdf['spinde'] = None


#getting 5s
wdf["sodi_poli_leaning_5"] = wdf['user_poli_leaning_num_sodi'].map(maps.f_poli_leaning_map)

#gettting 3s
wdf['claude_gpt_poli_leaning_3'] = wdf['claude_gpt_poli_leaning'].map(maps.poli_3_map)
wdf['gpt4o_gpt_poli_leaning_3'] = wdf['gpt4o_gpt_poli_leaning'].map(maps.poli_3_map)
wdf['allsides_leaning_3'] = wdf['allsides_leaning'].map(maps.poli_3_map)
wdf['sodi_poli_leaning_3'] = wdf['user_poli_leaning_num_sodi'].map(maps.f_poli_leaning_map)
wdf['sodi_poli_leaning_3'] = wdf['sodi_poli_leaning_3'].map(maps.poli_3_map)
#redoing source leaning - probably shouldn't do it here but have issues since not totally done coding
wdf['source_leaning'] = wdf['source'].map(maps.source_leaning_map)
wdf['source_leaning_3'] = wdf['source_leaning'].map(maps.poli_3_map)

#getting 2s
def is_biased(poli_leaning, overall_bias=None):

    poli_leaning = float(poli_leaning) if not (poli_leaning is None or pd.isna(poli_leaning)) else 0
    overall_bias = float(overall_bias) if not (overall_bias is None or pd.isna(overall_bias)) else 0

    if overall_bias == 0 or poli_leaning == 0:
        return False
    return True

wdf['claude_gpt_primary_bias_amnt_num'] = wdf['claude_gpt_primary_bias_amnt'].map(maps.bias_amount_map)
wdf['gpt4o_gpt_primary_bias_amnt_num'] = wdf['gpt4o_gpt_primary_bias_amnt'].map(maps.bias_amount_map)

wdf['claude_biased'] = wdf.apply(lambda x: is_biased(x['claude_gpt_poli_leaning_3'], x['claude_gpt_primary_bias_amnt_num']), axis=1)
wdf['gpt4o_biased'] = wdf.apply(lambda x: is_biased(x['gpt4o_gpt_poli_leaning_3'], x['gpt4o_gpt_primary_bias_amnt_num']), axis=1)
wdf['allsides_biased'] = wdf.apply(lambda x: is_biased(x['allsides_leaning_3']), axis=1)
wdf['sodi_biased'] = wdf.apply(lambda x: is_biased(x['sodi_poli_leaning_3'], x['user_bias_amnt_num_sodi']), axis=1)


#stances
#not sure we should call it this but could be neat
#note that there wont be any aireg, since can't tell that ahead of time.
wdf['source_article_subject'] = wdf['pull'].map(maps.pull_article_subject_map)
wdf['sodi_ai_stance'] = wdf['user_ai_leaning_num_sodi'].map(maps.f_ai_leaning_map)
wdf['sodi_aireg_stance'] = wdf['user_aireg_leaning_num_sodi'].map(maps.f_aireg_leaning_map)
wdf['sodi_imm_stance'] = wdf['user_imm_leaning_num_sodi'].map(maps.f_imm_leaning_map)

#TABLES

def get_stats(df, col1, col2):
    df = df[[col1, col2]].dropna() #can this work?
    if df.empty:
        print(f"No data available for columns {col1} and {col2}.")
        return {"col1": col1, "col2": col2, "accuracy": None, "agreement_count": None, "kappa": None, "num_rows": 0}
    accuracy = accuracy_score(df[col1], df[col2])
    agreement_count = (df[col1] == df[col2]).sum()
    kappa = cohen_kappa_score(df[col1], df[col2])
    return {"col1": col1, "col2": col2, "accuracy": accuracy, "agreement_count": agreement_count, "kappa": kappa, "num_rows": len(df)}

pack_threes = ["claude_gpt_poli_leaning_3", "gpt4o_gpt_poli_leaning_3", "allsides_leaning_3", "sodi_poli_leaning_3", "source_leaning_3", "baly"]
pack_fives = ["claude_gpt_poli_leaning", "gpt4o_gpt_poli_leaning", "allsides_leaning", "sodi_poli_leaning_5", "source_leaning"] #no emmet yet
pack_twos = ['claude_biased', 'gpt4o_biased', 'allsides_biased', 'sodi_biased', 'spinde']
pack_bases = ["claude_gpt_article_subject", "gpt4o_gpt_article_subject", "user_article_subject_sodi", "source_article_subject"]
pack_stance_ai = ["claude_gpt_ai_leaning", "gpt4o_gpt_ai_leaning", "sodi_ai_stance"]
pack_stance_aireg = ["claude_gpt_aireg_leaning", "gpt4o_gpt_aireg_leaning", "sodi_aireg_stance"]
pack_stance_imm = ["claude_gpt_imm_leaning", "gpt4o_gpt_imm_leaning", "sodi_imm_stance"] 

allsets = {
    "pack_threes": pack_threes,
    "pack_fives": pack_fives,
    "pack_twos": pack_twos,
    "pack_bases": pack_bases,
    "pack_stance_ai": pack_stance_ai,
    "pack_stance_aireg": pack_stance_aireg,
    "pack_stance_imm": pack_stance_imm
}
big_agreements = []
for pack_name, pack_columns in allsets.items():
    for i in range(len(pack_columns)):
        for j in range(i + 1, len(pack_columns)):
            col1 = pack_columns[i]
            col2 = pack_columns[j]
            stats = get_stats(wdf, col1, col2)
            stats['pack'] = pack_name
            big_agreements.append(stats)

big_agreements_df = pd.DataFrame(big_agreements)
big_agreements_df.to_csv('big_agreements.csv', index=False)
