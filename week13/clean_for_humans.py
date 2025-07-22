import pandas as pd
import re

df = pd.read_csv('./full_study_files/set1_full_raw.csv')

dfai = pd.read_csv('./full_study_files/set1_sampled400_gpt4o.csv')
df.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
dfai.rename(columns={'Unnamed: 0': 'orig_ID'}, inplace=True)

dfh = dfai.copy()
dfh['set_number'] = 'set1'
dfh['rater_email'] = 'sodikroehler@gmail.com'
dfh['human_leaning'] = ''
dfh['human_retweet'] = ''
dfh['human_subject'] = ''
dfh['human_agree_with'] = ''
dfh['coding_date'] = pd.to_datetime('1990-01-01')
dfh['coding_time'] = pd.to_datetime('00:00:00').time()

dfh['text'] = dfh['text'].str.replace(r'\n\n', '\n', regex=True)
identifying_words = ['NPR', 'BBC', 'Reuters', 'AP', 'Associated Press', 'The Guardian', 
                     'The New York Times', 'MSNBC', 'Fox News', 'Daily Caller', 'Washington Examiner', 
                     'Consider This', 'Short Wave', 'The Daily', 'News Hour', 'Al Jazeera', 'Xinhua',]

DCNF
dailycallernewsfoundation

pattern = '|'.join(map(re.escape, identifying_words))   
dfh['text'] = (
    dfh['text']
    .str.replace(pattern, '[SOURCE]', regex=True)
    .str.replace(r'\s{2,}', '[SOURCE]', regex=True)
    .str.strip()
)
dfh['title'] = (
    dfh['title']
    .str.replace(pattern, '[SOURCE]', flags=re.IGNORECASE, regex=True)
    .str.replace(r'\s{2,}', '[SOURCE]', regex=True)
    .str.strip()
)

dfh_em = dfh.copy()
dfh_em['rater_email'] = 'Emmet.mathieu@gmail.com'
dfh = pd.concat([dfh, dfh_em], ignore_index=True)
dfh['ID'] = range(1, len(dfh) + 1)

dfai.to_csv('./full_study_files/set1_sampled400_gpt4o.csv', index=False)
dfh.to_csv('./full_study_files/set1_sampled400_forHumans.csv', index=False)
