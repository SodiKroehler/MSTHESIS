import pandas as pd
#scrappy little test set:
dfm0 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/combined_gdelt_unfiltered2.csv')
#The first dataset pulled, filtered on url keyword (without deepseek):
dfm1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/combined_gdelt_ai.csv')
#pull_shards, which cost 300, created this file. this should be from the original bigquery query, and may not include all rows that were pulled form this. also isn't scraped
#not use pass2, which only has a few rows but was probalby from immigration
dfjn1_raw = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/june_revisit/raw_ai_pass_1.csv', sep='\t', encoding='utf-8')
#this was combined with some other files in to dfunk1

#we don't actually know what script generated this, but it was probably the week13 script that got overwritten horribly somehow. people that don't use git are the worst. this means i am the worst.
#this probably has june and july in it, so we should check and verify
df_unk1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/raw_july_combinedimmandai_full.csv')
#and then scraper.py in june_revisit parsed this into the below:
dfj1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/june_revisit/cleaned_ai_pass_1.csv')
#then dfm0, dfm1, dfjn1 were combined into the following in filter_previous_pulls.py:
dfjj1 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/june_revisit/combined_gdelt_filtered.csv')
#now we finally get to combine_july_with_june_pulls:
#it took in dfunk1 (which coincidentally was passed thorugh the scraper with the same name), dfjj1 and made: 
dfj2 = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/july_revisit/final_july_combined_immandai.csv')
#this was used in both weeks to retrieve samples



#________________________OBSERVATIONS________________________
#after verification, df_unk1 only have july data. so it got combined, and then pushed through the scraper with the same name. it has no june data or before until dfjj1

#filter previous pulls is where there may be issues. 
#it says 1498 in march, 402 in unfilterd march, 738 in pass 1, with no july.
#dfj2 says 1498 in march, 219 in unfiltered march, 719 in pass 1, with 3482 in julyimm, and 236 in julyai

#dfunk2 has 3495 for imm and 890 for ai.
#so we went from 115533 in march 1 + 34975 from march0 to have total of 115533 + 3495 + 890 = 119918 in march. filtered this down to 1498 + 402 based on source/na/stuff, should give us 1900 total march

#we're missing 890 - 236 in julyai, 402 - 219 in march0, and then 3495 - 3482 in julyimm
