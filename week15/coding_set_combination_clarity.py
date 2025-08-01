import pandas as pd
import re
import official_maps as maps

#_________________________________CODING________________________________________

#Set 1 - coded
df_jul8f = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/MSTHESIS_BACKEND_SHEET-completedJuly8.csv')
df_jul8f.columns = ['index', 'title', 'text', 'date', 'source_url', 'pull', 'source',
       'source_leaning', 'clean', 'gpt_leaning', 'LLM_RESPONSE', 'set_number',
       'rater_email', 'user_poli_leaning', 'user_agree_with', 'user_article_subject',
       'user_wants_to_see', 'coding_date', 'coding_time', 'ID']
# #orig: ['index', 'title', 'text', 'date', 'article_url', 'pull', 'source',
#        'source_leaning', 'clean', 'gpt_leaning', 'LLM_RESPONSE', 'set_number',
#        'rater_email', 'human_leaning', 'human_retweet', 'human_subject',
#        'human_agree_with', 'coding_date', 'coding_time', 'ID']

for col in ['user_ai_leaning', 'user_aireg_leaning', 'user_imm_leaning', 'user_bias_amnt', 'user_bias_cause', 'gkg_id']:
    df_jul8f[col] = None
df_jul8f['set_number'] = 'set1'
#even though we're dedpulicating good later, we don't want to include non-coded rows
df_jul8 = df_jul8f.drop_duplicates(subset=['source_url']).dropna(subset=['source_url', 'user_poli_leaning']) #always has poli leaning if coded at this time



#Set 2 -coded
df_jul16_full = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week14/MSTHESIS_BACKEND_SHEET - PRODUCTION_jul16.csv')
df_jul16_full.rename(columns={
    'article_url': 'source_url',
}, inplace=True)
df_jul16_full['set_number'] = 'set2'
df_jul16 = df_jul16_full.drop_duplicates(subset=['source_url']).dropna(subset=['source_url', 'user_poli_leaning']) #same here i think


#Set 3 - coded
df_jul24_full = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/MSTHESIS_BACKEND_SHEET - PRODUCTION - pulled jul24.csv')
df_jul24_full.rename(columns={
    'article_url': 'source_url',
}, inplace=True)
df_jul24_full['set_number'] = 'set3'
df_jul24 = df_jul24_full.drop_duplicates(subset=['source_url']).dropna(subset=['source_url', 'user_poli_leaning']) #same here i think

#Set 4
#current state is coding
df_jul25 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/set4_sampled_rows.csv')
for col in ['user_poli_leaning', 'user_agree_with', 'user_wants_to_see', 
            'user_article_subject', 'user_ai_leaning', 'user_aireg_leaning', 
            'user_imm_leaning', 'user_bias_amnt', 'user_bias_cause', 'gpt_leaning', 'coding_date', 'coding_time']:
    if col not in df_jul25.columns:
        df_jul25[col] = None

df_jul25['set_number'] = 'set4'
#don't need to drop any here, as we're not done coding

#as of 6:30 on jan 30, we're not using this sampler any more as i think its giving us issues.
#here's what we have coded so far
jul_30 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/MSTHESIS_BACKEND_SHEET - PRODUCTION - pulledjul30 - midR4.csv')
#well keep out everything with isCoded = True, as we know its good
df_jul25_true = jul_30[jul_30['isCoded'] == True].copy()
#then it makes sense to keep what we've coded in previous sets, so
df_jul25_in_set1 = df_jul25_true[df_jul25_true['set_number'] == 'set1'].copy()
df_jul25_in_set2 = df_jul25_true[df_jul25_true['set_number'] == 'set2'].copy()
df_jul25_in_set3 = df_jul25_true[df_jul25_true['set_number'] == 'set3'].copy()
#we've already fitlered those if they weren't coded, so we should be good to combine all these sets
df_jul30 = pd.concat([df_jul25_in_set1, df_jul25_in_set2, df_jul25_in_set3], ignore_index=True)

#we can see after running that this removed all of the allsides, which we definitely need
#well get these from the old set4, since we know they were all there:
df_jul30_allsides_1 = df_jul25[df_jul25['pull'].str.startswith('df7')].copy()
df_jul30_allsides_1 = df_jul30_allsides_1.drop_duplicates(subset=['source_url']).dropna(subset=['source_url'])
df_jul30_allsides_2 = df_jul25[df_jul25['pull'].str.startswith('df6')].copy()
df_jul30_allsides_2 = df_jul30_allsides_2.drop_duplicates(subset=['source_url']).dropna(subset=['source_url'])

df_jul30 = pd.concat([df_jul30, df_jul30_allsides_1, df_jul30_allsides_2], ignore_index=True)

#we know that all these were in this pull, so they have chatgpt and claude rankings. we'll still go through the rest of the process now though, to make sure we keep as much from previous sets as possible.
#to avoid confusion, we'll make this a new set
df_jul25['set_number'] = 'set5'


abomindable_dictionary = {}

cwfdf = pd.concat([df_jul8, df_jul16, df_jul24, df_jul30], ignore_index=True)

gpt_leaning_name = f"{cwfdf.loc[0, 'set_number']}_gpt_leaning"
cwfdf['user_poli_leaning_num'] = cwfdf["user_poli_leaning"].map(maps.poli_leaning_map).astype(float)
cwfdf['user_agree_with'] = cwfdf['user_agree_with'].astype(bool) #just t/f
cwfdf['user_article_subject'] = cwfdf['user_article_subject'].astype(str) #missing ai reg here, but otherwise the same
cwfdf['user_wants_to_see_num'] = cwfdf['user_wants_to_see'].map(maps.want_to_see_map).astype(float)
cwfdf['user_ai_leaning_num'] = cwfdf['user_ai_leaning'].map(maps.ai_leaning_map).astype(float)
cwfdf['user_aireg_leaning_num'] = cwfdf['user_aireg_leaning'].map(maps.aireg_leaning_map).astype(float)
cwfdf['user_imm_leaning_num'] = cwfdf['user_imm_leaning'].map(maps.imm_leaning_map).astype(float)
cwfdf['user_bias_amnt_num'] = cwfdf['user_bias_amnt'].map(maps.bias_amount_map).astype(float)
cwfdf['user_bias_cause'] = cwfdf['user_bias_cause'].astype(str).fillna('')  #also just leaving the same here too
cwfdf[gpt_leaning_name] = cwfdf['gpt_leaning'].map(maps.poli_leaning_map).astype(float)
cwfdf['source_leaning'] = cwfdf['source'].map(maps.source_leaning_map).astype(str).fillna('')
cwfdf['source_leaning'] = cwfdf['source_leaning'].map(maps.poli_leaning_map).astype(float)
cwfdf['rater'] = cwfdf['rater_email'].map(maps.rater_map).astype(str).fillna('')
cwfdf['gpt_leaning'] = "didn't keep this"

for arti in cwfdf['source_url'].unique():
    abomindable_dictionary[arti] = {
        'user_poli_leaning_num_sodi': None,
        'user_poli_leaning_num_emmet': None,
        'user_agree_with_sodi': None,
        'user_agree_with_emmet': None,
        'user_article_subject_sodi': None,
        'user_article_subject_emmet': None,
        'user_wants_to_see_num_sodi': None,
        'user_wants_to_see_num_emmet': None,
        'user_ai_leaning_num_sodi': None,
        'user_ai_leaning_num_emmet': None,
        'user_aireg_leaning_num_sodi': None,
        'user_aireg_leaning_num_emmet': None,
        'user_imm_leaning_num_sodi': None,
        'user_imm_leaning_num_emmet': None,
        'user_bias_amnt_num_sodi': None,
        'user_bias_amnt_num_emmet': None,
        'user_bias_cause_sodi': '',
        'user_bias_cause_emmet': '',
        'source_leaning': None,
        'coding_date_sodi': None,
        'coding_date_emmet': None,
        'coding_time_sodi': None,
        'coding_time_emmet': None,
        'gpt_leaning': None,
        'pull': None,
        'source': None,
        'title': None,
        'text': None,
        'date': None,
        'gkg_id': None,
        'set_number': None,
        'source_url': arti,
    }
value_coded_in_other_sets = {} #is 101 long, so real problem

def allow_write(row, rowValueKey, dictValueKey):

    user_cols = ['user_poli_leaning', 'user_agree_with',
       'user_article_subject', 'user_wants_to_see', 'coding_date',
       'coding_time', 'user_ai_leaning', 'user_aireg_leaning',
       'user_imm_leaning', 'user_bias_amnt', 'user_bias_cause', 'isCoded',
       'user_poli_leaning_num', 'user_wants_to_see_num', 'user_ai_leaning_num',
       'user_aireg_leaning_num', 'user_imm_leaning_num', 'user_bias_amnt_num']
    isAboutIntendedRater = dictValueKey.endswith('_' + row.rater) and rowValueKey in user_cols
   
    #first we check if the thing is worth writing
    if getattr(row, rowValueKey) is None or pd.isna(getattr(row, rowValueKey)):
        return False
    
    if not isAboutIntendedRater:
        #if its not about the intended rater, we can immediately reject it
        return False

    if abomindable_dictionary[row.source_url]['set_number'] is None:
        #nothign has happened yet, write away
        return True
    elif row.set_number == abomindable_dictionary[row.source_url]['set_number']:
        #we're currently coding this set, no worries here
        return True
    elif abomindable_dictionary[row.source_url][dictValueKey] is None:
        #already coded in a different set, but somehow the value is still None
        # abomindable_dictionary[row.source_url]["values_coded_in_other_sets"][valueKey] = row.set_number - we dont write here, just return true
        #but we'll also record it jic
        if row.source_url not in value_coded_in_other_sets:
            value_coded_in_other_sets[row.source_url] = {}
            value_coded_in_other_sets[row.source_url][dictValueKey] = [row.set_number]
        else:
            if dictValueKey not in value_coded_in_other_sets[row.source_url]:
                value_coded_in_other_sets[row.source_url][dictValueKey] = [row.set_number]
            else:
                if row.set_number not in value_coded_in_other_sets[row.source_url][dictValueKey]:
                    value_coded_in_other_sets[row.source_url][dictValueKey].append(row.set_number)
        return True
    return False

for row in cwfdf.itertuples():
    if allow_write(row, 'user_poli_leaning_num', 'user_poli_leaning_num_' + row.rater):
        abomindable_dictionary[row.source_url]['user_poli_leaning_num_' + row.rater] = row.user_poli_leaning_num
        abomindable_dictionary[row.source_url]['coded'] = True
    # if row.source_url == 'https://www.npr.org/2025/04/14/nx-s1-5323918/self-deportation-immigration-trump':
    #     breakpoint()
    if allow_write(row, 'user_agree_with', 'user_agree_with_' + row.rater):
        abomindable_dictionary[row.source_url]['user_agree_with_' + row.rater] = row.user_agree_with
    if allow_write(row, 'user_article_subject', 'user_article_subject_' + row.rater):
        abomindable_dictionary[row.source_url]['user_article_subject_' + row.rater] = row.user_article_subject
    if allow_write(row, 'user_wants_to_see_num', 'user_wants_to_see_num_' + row.rater):
        abomindable_dictionary[row.source_url]['user_wants_to_see_num_' + row.rater] = row.user_wants_to_see_num
    if allow_write(row, 'user_ai_leaning_num', 'user_ai_leaning_num_' + row.rater):
        abomindable_dictionary[row.source_url]['user_ai_leaning_num_' + row.rater] = row.user_ai_leaning_num
    if allow_write(row, 'user_aireg_leaning_num', 'user_aireg_leaning_num_' + row.rater):
        abomindable_dictionary[row.source_url]['user_aireg_leaning_num_' + row.rater] = row.user_aireg_leaning_num
    if allow_write(row, 'user_imm_leaning_num', 'user_imm_leaning_num_' + row.rater):
        abomindable_dictionary[row.source_url]['user_imm_leaning_num_' + row.rater] = row.user_imm_leaning_num
    if allow_write(row, 'user_bias_amnt_num', 'user_bias_amnt_num_' + row.rater):
        abomindable_dictionary[row.source_url]['user_bias_amnt_num_' + row.rater] = row.user_bias_amnt_num
    if allow_write(row, 'user_bias_cause', 'user_bias_cause_' + row.rater):
        abomindable_dictionary[row.source_url]['user_bias_cause_' + row.rater] = str(row.user_bias_cause).strip()
    if allow_write(row, 'gpt_leaning', 'gpt_leaning'):
        abomindable_dictionary[row.source_url]['gpt_leaning'] = row.gpt_leaning
    if allow_write(row, 'source_leaning', 'source_leaning'):
        abomindable_dictionary[row.source_url]['source_leaning'] = row.source_leaning
    if allow_write(row, 'coding_date', 'coding_date_' + row.rater):
        abomindable_dictionary[row.source_url]['coding_date_' + row.rater] = row.coding_date
    if allow_write(row, 'coding_time', 'coding_time_' + row.rater):
        abomindable_dictionary[row.source_url]['coding_time_' + row.rater] = row.coding_time
    if allow_write(row, 'pull', 'pull'):
        abomindable_dictionary[row.source_url]['pull'] = row.pull
    if allow_write(row, 'source', 'source'):
        abomindable_dictionary[row.source_url]['source'] = row.source
    if allow_write(row, 'title', 'title'):
        abomindable_dictionary[row.source_url]['title'] = row.title
    if allow_write(row, 'text', 'text'):
        abomindable_dictionary[row.source_url]['text'] = row.text
    if allow_write(row, 'date', 'date'):
        abomindable_dictionary[row.source_url]['date'] = row.date
    if allow_write(row, 'gkg_id', 'gkg_id'):
        abomindable_dictionary[row.source_url]['gkg_id'] = row.gkg_id
    if row.set_number is not None:
        if abomindable_dictionary[row.source_url]['set_number'] is None:
            #we can (possibly temporarily) set the set_number in this case
            abomindable_dictionary[row.source_url]['set_number'] = row.set_number
        elif abomindable_dictionary[row.source_url]['set_number'] is not None and row.user_poli_leaning_num is not None:
            #we put it wrong before, we'll fix it now
            abomindable_dictionary[row.source_url]['set_number'] = row.set_number
        #otherwise, it wasn't "coded" in this set, we can leave it for the one it was coded in


 

combined_wide_form = pd.DataFrame.from_dict(abomindable_dictionary, orient='index')
#set a custom id for each article/set combination - there'll be no duplicates by this point
combined_wide_form['article_set_unique_id'] = [id for id in range(1, len(combined_wide_form) + 1)]
#convention has been to leave 1k between sets, just to be safe and for ease of use
combined_wide_form['article_set_unique_id'] = combined_wide_form.apply(lambda x: x['article_set_unique_id'] + (int(x['set_number'][-1]) * 1000), axis=1)
#we also make it a string, sometimes that helps avoid confusion
combined_wide_form['article_set_unique_id'] = combined_wide_form['set_number'].astype(str) + '_' + combined_wide_form['article_set_unique_id'].astype(str)


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

#keeping only what's already coded, and set4
# long_df = long_df[(long_df['set_number'] == "set4") | (long_df["coded"] == True)].copy()

#choosing to replace set 3 id because i dont like it
long_df['set3_ID'] = range(1, len(long_df) + 1)
long_df['set3_ID'] = long_df['set3_ID'] + (long_df['set_number'].str[-1].astype(int) * 1000)
# long_df.loc[long_df['pull'].str.startswith('df7'), 'set3_ID'] += 10000 #ughhh but have to
# long_df['set3_ID'] = long_df['set3_ID'].fillna(long_df['set4_id'])
# long_df.drop(columns=['set4_id'], inplace=True)

# #reverse the num labels
numcols = ['user_poli_leaning_num', 'user_wants_to_see_num', 'user_ai_leaning_num', 
            'user_aireg_leaning_num', 'user_imm_leaning_num', 'user_bias_amnt_num']



flipped_maps = [

    maps.f_poli_leaning_map,
    maps.f_want_to_see_map,
    maps.f_ai_leaning_map,
    maps.f_aireg_leaning_map,
    maps.f_imm_leaning_map,
    maps.f_bias_amount_map
]
for idx, col in enumerate(numcols):
    if col in long_df.columns:
        new_col_name = col.replace('_num', '')
        long_df[new_col_name] = long_df[col].astype(float).map(flipped_maps[idx])
        long_df.drop(columns=[col], inplace=True)

long_df['rater_email'] = long_df['rater'].map(maps.f_rater_map)
long_df.rename(columns={"source_url": "article_url"}, inplace=True)

junk_cols = ['timestamp', 'themes', 'DF_IDX', 'url', 'url_tuple', 'URL_new', 'images', 'authors', 'movies', 'tags', 'clean', 'LLM_RESPONSE', 'old_bad_id', 'ID'] 
for c in junk_cols:
    long_df[c] = None

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

long_df.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_long_form_jul30.csv', index=False)

#somehow getting extra rows
combined_wide_form = combined_wide_form[combined_wide_form['source_url'].isin(long_df['article_url'])].copy()
combined_wide_form.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/combined_wide_form_jul30.csv', index=False)


#edit jul30 - we added rows to set4, but don't want to merge whats currently coded with them. so we'll split off the new rows here, and manually paste into the excel.
#the creations of this file will be the same as beforehand, just with the extra rows. after coding, we should be able to update the entire thing, with no issues.
# pulljul30 = pd.read_csv('MSTHESIS_BACKEND_SHEET - PRODUCTION - pulledjul30 - midR4.csv')
# manual_adds = long_df[~long_df['article_url'].isin(pulljul30['article_url'])].copy()
# manual_adds = manual_adds[manual_adds['pull'].str.startswith('df7')].copy()
# manual_adds.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/delta_lines_from_manualallsides_pull_to_add_to_production_sheet.csv', index=False)



#has 150 rows not in long_df rn