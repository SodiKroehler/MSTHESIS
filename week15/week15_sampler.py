import pandas as pd
import re

cmbdf = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/complete_all_articles_jul25.csv')


# #shorts has this for pull counts: pull
# ai_march            224
# ai_pass_1           145
# unfiltered_march     52

#we have a total of 421 at some level of coding, we can complete what we hvae already done.
#then last add all the allsides from our approved sources, and 50 from each july pull
#then we'll sample some more from allsides until we get a nice round number
approved_sources = [
  'nytimes.com', 'msnbc.com', 'theguardian.com',
  'reuters.com', 'apnews.com', 'npr.org',
  'foxnews.com', 'dailycaller.com', 'washingtonexaminer.com',
  'bbc.com', 'aljazeera.com', 'english.news.cn', 'bbc.co.uk',


  #had to add for manual pull - these are diff names but still same sources (handle later??)
  'foxbusiness.com'
] 

cmbdf['source'] = cmbdf['source_url'].str.lower().str.split('/').str[2].str.replace('www.', '', regex=False)
approved_as = cmbdf[cmbdf['source'].isin(approved_sources) & cmbdf['pull'].str.contains('df6')].copy()#jul25_allsides

#on jul 30, we pulled another set of articles, so adding thtat too:
manual_pull = cmbdf[cmbdf['source'].isin(approved_sources) & cmbdf['pull'].str.contains('df7')].copy()#jul30_allsides
#this would make the below untrue, but 50 rows makes sense and i dont want to have to figure out which to delete. since we have random state, we're leaving it

#found out that 50 give us a total of 622, so lets sample 78 more from the july and june, or 37 more from each, so 87
#edit jul28- we were sampling 89 from allsides dataset, which makes no sense. also, we only have 101 from allsides, but want to get more, so well just sample 50 from each of the july pulls
july_imm = cmbdf[cmbdf['pull'] == 'df4'].sample(n=50, random_state=42)
july_ai = cmbdf[cmbdf['pull'] == 'df5'].sample(n=50, random_state=42)



jul25 = pd.concat([approved_as, manual_pull, july_imm, july_ai], ignore_index=True)
jul25['set_number'] = 'set4'
em = jul25.copy()
em['rater_email'] = "emmet.mathieu@gmail.com"
jul25['rater_email'] = "sodi.kroehler@gmail.com"
jul25 = pd.concat([jul25, em], ignore_index=True)
jul25.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/set4_sampled_rows.csv', index=False)






# for col in previous_longers.columns:
#     if col not in jul25.columns:
#         jul25[col] = None

# # max_id = previous_longers['set3_ID'].fillna('').astype(str).split('_').str[-1].astype(int).max()


# # jul25['set3_ID'] = range(1, len(jul25) + 1)
# # jul25['set3_ID'] = range(max_id+1, max_id+1 + len(jul25))
# # jul25['set3_ID'] = "set4_" + jul25['set3_ID'].astype(str)
# fin_25 = pd.concat([previous_longers, jul25], ignore_index=True)

# fin_25['set4_ID'] = range(1, len(fin_25) + 1)
# fin_25['set4_ID'] = lambda x: x['set4_ID'] + (x['set_number'].str[-1].astype(int) * 1000)
# fin_25['set4_ID'] = fin_25['set_number'] + '_' + fin_25['set4_ID'].astype(str)
# fin_25 = fin_25['set3_ID'].fillna(fin_25['set4_ID'])












# shorts = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_shorts.csv')
# coded = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_coded.csv')
# dfjas = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/ALLSIDES/ALLSIDES/final_jul25allsides.csv')
# previous_longers = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_long_form_pre_jul25.csv')
# previous_longers = previous_longers.drop_duplicates(subset=['source_url'], keep='last').dropna(subset=['source_url', 'user_poli_leaning']) #always has poli leaning

# full_prev = shorts.merge(coded,on=['article_set_unique_id'],how='outer')
# if not full_prev.shape[0] == shorts.shape[0]:
#     print("Warning: Merging shorts and coded resulted in a different number of rows than shorts alone.")


# max_id = shorts['article_set_unique_id'].str.split('_').str[-1].astype(int).max()
# jul25['article_set_unique_id'] = range(max_id+1, max_id+1 + len(jul25))
# jul25['article_set_unique_id'] = "set4_" + jul25['article_set_unique_id'].astype(str)
# #['source_url', 'pull', 'title', 'text', 'date', 'gkg_id', 'set_coded_in', 'article_set_unique_id']
# for col in jul24backend.columns:
#     if col not in jul25.columns:
#         jul25[col] = None
#     if col not in full_prev.columns:
#         full_prev[col] = None

#         #note this is missing allsides_bias, but we're not writing back to coded or shorts until its actually coded

# fin_25 = pd.concat([full_prev, jul25], ignore_index=True)


# fin_25['set_3_id'] = fin_25['article_set_unique_id'].str.split('_').str[-1].astype(int)+4000
# fin_25.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/MSTHESIS_BACKEND_SHEET - PRODUCTION - for jul25.csv', index=False)

# # fjul25c = fjul25.copy()
# # fjul25c['allside_bias'] = None
# # fjul25c = fjul25c.merge(
# #     dfjas[['source_url', 'allside_bias']].drop_duplicates(subset='source_url', keep='first'),
# #     on='source_url',
# #     how='left'
# # )

# # for col in coded.columns:
# #     if col not in fjul25c.columns:
# #         fjul25c[col] = None

# # coded['allsides_bias'] = None
# # coded = pd.concat([coded, fjul25c], ignore_index=True)