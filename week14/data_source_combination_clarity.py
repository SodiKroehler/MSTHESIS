import pandas as pd
import re


#scrappy little test set:
dfm0 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/combined_gdelt_unfiltered2.csv')
#The first dataset pulled, filtered on url keyword (without deepseek):
dfm1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/combined_gdelt_ai.csv')
#pull_shards, which cost 300, created this file. this should be from the original bigquery query, and may not include all rows that were pulled form this. also isn't scraped
#not using pass2, which only has a few rows but was probalby from immigration
dfjn1_raw = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/june_revisit/raw_ai_pass_1.csv', sep='\t', encoding='utf-8')
#this was combined with some other files in to dfunk1

#we don't actually know what script generated this, but it was probably the week13 script that got overwritten horribly somehow. people that don't use git are the worst. this means i am the worst.
#this probably has june and july in it, so we should check and verify

#wrong to above, this just has july data, and was formed with parse_raw_gdelt_bigquery from these two files which im adding now (jul 23)
df_all_ai_july = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/all_ai_rows.csv')
df_all_imm_july = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/all_immigration_rows.csv')
df_unk1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/raw_july_combinedimmandai_full.csv')
#and then scraper.py in june_revisit parsed this into the below:
dfj1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/june_revisit/cleaned_ai_pass_1.csv')
#then dfm0, dfm1, dfjn1 were combined into the following in filter_previous_pulls.py:
dfjj1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/june_revisit/combined_gdelt_filtered.csv')
#now we finally get to combine_july_with_june_pulls:
#it took in dfunk1 (which coincidentally was passed thorugh the scraper with the same name), dfjj1 and made: 
dfj2 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/final_july_combined_immandai.csv')
#this was used in both weeks to retrieve samples


#allsides jul25
dfjas = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/ALLSIDES/ALLSIDES/final_jul25allsides.csv')

#_________________________COMBINE EVERYTHING SEPERATELY________________________
#first get a seperate df for each pull
#DATASET 1
df1 = dfm0.drop_duplicates(subset=['article_url']).dropna(subset=['article_url']).copy()
df1.columns = ['randomid', 'title', 'text', 'date', 'source_url']
df1['pull'] = 'df1'
df1['gkg_id'] = 'unk'
df1 = df1[['source_url', 'pull', 'title', 'text', 'date', 'gkg_id']]

#DATASET 2
df2 = dfm1.drop_duplicates(subset=['article_url']).dropna(subset=['article_url']).copy()
df2.columns = ['randomid', 'title', 'text', 'date', 'source_url']
df2['pull'] = 'df2'
df2['gkg_id'] = 'unk'
df2 = df2[['source_url', 'pull', 'title', 'text', 'date', 'gkg_id']]

#DATASET 3
df3 = dfjn1_raw.copy()
df3.columns = ['dateish_thing', 'gkg_id', 'source', 'source_url', 'v2_themes', 'locations', 'persons', 'orgs', 'gcam', 'amnts', 'extras']
df3['pull'] = 'df3'
df3 = df3.drop_duplicates(subset=['source_url']).dropna(subset=['source_url'])
merging_unk0 = dfj1[['URL_new', 'TITLE', 'TEXT', 'DATE_new']]
merging_unk0.columns = ['source_url', 'title', 'text', 'date']
merging_unk0 = (
    merging_unk0
    .groupby('source_url', as_index=False)
    .agg({
        'title': 'first',
        'text': 'first',
        'date': 'first'
    })
)
df3 = df3.merge(merging_unk0, on='source_url', how='left')
df3 = df3[['source_url', 'pull', 'title', 'text', 'date', 'gkg_id']]

#DATASET 4
df4 = df_all_ai_july.drop_duplicates(subset=['DocumentIdentifier']).dropna(subset=['DocumentIdentifier']).copy()
df4['pull'] = 'df4'
df4.columns = ['gkg_id', 'gkg_timestamp', 'source_common_name', 'source_url', 'v2themes', 'pull']
merging_unk1 = df_unk1[['ARTICLE_URL', 'TITLE_new', 'TEXT_new', 'DATE',]] #didn't filter bc we shoulldn't need to if we do left join
merging_unk1.columns = ['source_url', 'title', 'text', 'date']
merging_unk1 = (
    merging_unk1
    .groupby('source_url', as_index=False)
    .agg({
        'title': 'first',
        'text': 'first',
        'date': 'first'
    })
)
df4 = df4.merge(merging_unk1, on='source_url', how='left')
df4 = df4[['source_url', 'pull', 'title', 'text', 'date', 'gkg_id']]

#DATASET 5
df5 = df_all_imm_july.drop_duplicates(subset=['DocumentIdentifier']).dropna(subset=['DocumentIdentifier']).copy()
df5['pull'] = 'df5'
df5.columns = ['gkg_id', 'gkg_timestamp', 'source_common_name', 'source_url', 'v2themes', 'pull']
# merging_unk2 = df_unk1[['GKG_ID', 'ARTICLE_URL', 'TITLE_new', 'TEXT_new', 'DATE', 'PULL']] #didn't filter bc we shoulldn't need to if we do left join
# merging_unk2.columns = ['gkg_id', 'source_url', 'title', 'text', 'date', 'pull']
df5 = df5.merge(merging_unk1, on='source_url', how='left')
df5 = df5[['source_url', 'pull', 'title', 'text', 'date', 'gkg_id']]


#DATASET 6
df6 = dfjas[['source_url', 'pull', 'title', 'text', 'date', 'gkg_id']].copy()
df6[df6['pull'].str.endswith('_ai')]['pull'] = 'df6_ai'
df6[df6['pull'].str.endswith('_imm')]['pull'] = 'df6_imm'
# cdf.columns = ['source_url', 'pull', 'title', 'text', 'date', 'gkg_id', 'allside_bias', 'human_clean_text', 'human_clean_title']


cmbdf = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
pull_priority = {'df1': 1, 'df2': 2, 'df3': 3, 'df4': 4, 'df5': 5, 'df6_ai': 0, 'df6_imm': 0}
cmbdf_dedup = (
    cmbdf.assign(pull_rank=cmbdf['pull'].map(pull_priority))
    .sort_values('pull_rank', ascending=False)
    .drop_duplicates(subset=['source_url'], keep='first')
    # .drop(columns=['pull_rank'])
)

# issues_in_df4 = df4[~df4['source_url'].isin(cmbdf_dedup[cmbdf_dedup['pull'] == "df4"]['source_url'])] 
# df4[~df4['source_url'].isin(df5['source_url'])]


 
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
cmbdf_dedup['human_clean_text'] = cmbdf_dedup['text'].apply(clean_for_humans)
cmbdf_dedup['human_clean_title'] = cmbdf_dedup['title'].apply(clean_for_humans)
cmbdf_dedup.to_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/week15/complete_all_articles_jul25.csv', index=False)



#________________________OBSERVATIONS________________________
#after verification, df_unk1 only have july data. so it got combined, and then pushed through the scraper with the same name. it has no june data or before until dfjj1

#filter previous pulls is where there may be issues. 
#it says 1498 in march, 402 in unfilterd march, 738 in pass 1, with no july.
#dfj2 says 1498 in march, 219 in unfiltered march, 719 in pass 1, with 3482 in julyimm, and 236 in julyai

#dfunk2 has 3495 for imm and 890 for ai.
#so we went from 115533 in march 1 + 34975 from march0 to have total of 115533 + 3495 + 890 = 119918 in march. filtered this down to 1498 + 402 based on source/na/stuff, should give us 1900 total march

#we're missing 890 - 236 in julyai, 402 - 219 in march0, and then 3495 - 3482 in julyimm

# >>> df_unk1[df_unk1['URL_new'] != df_unk1['URL']].shape
# (0, 20)