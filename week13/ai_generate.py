from cardui import Machina, Structura, Batcher
import pandas as pd

machin = Machina("OpenAI", "gpt-4o")
machin.auth()
machin.max_retries = 1


source_leanings = {
    "nytimes.com": "left",
    "msnbc.com": "left",
    "theguardian.com": "left",
    "reuters.com": "center",
    "apnews.com": "center",
    "npr.org": "center",
    "foxnews.com": "right",
    "dailycaller.com": "right",
    "washingtonexaminer.com": "right",
    "bbc.com": "global",
    # "bbc.co.uk": "global", not including in the graph as is bbc
    "aljazeera.com": "global",
    "english.news.cn": "global",
}

# df = pd.read_csv('../raw/GDELT/june_revisit/combined_gdelt_filtered.csv')
# df['source_leaning'] = df['source'].map(source_leanings)
# df['clean'] = 'HEADLINE: ' + df['title'] + '\n\n' + 'STORY TEXT: ' + df['text']
# df['gpt_leaning'] = None
# df = df.head()

#set 2 ____________________________________
df = pd.read_csv('full_study_files/set2_sampled.csv')
df['source_leaning'] = df['source'].map(source_leanings)
df['clean'] = 'HEADLINE: ' + df['title'] + '\n\n' + 'STORY TEXT: ' + df['text']
df = df[df['gpt_leaning'].isna()]
df_others = df[df['gpt_leaning'].isna() != True]
df = df.head(1)


struct1 = Structura()
struct1.PROMPT = "for each of the following headlines, determine if it is left-leaning (\"left\"), right-leaning (\"right\"), centrist with hardly any bias (\"center\") or not a polarized topic/could not be biased (\"none\"). Determine your answer, wait and think and see if it can be improved. Your answer for each should be only one word. #MY_INPUT_PLACEHOLDER"
struct1.INPUT_OBJECT_PLACEHOLDER = "#MY_INPUT_PLACEHOLDER"
struct1.INPUT_OBJECT_NAME = "stories"
struct1.INPUT_COLUMN_NAMES = ["clean"]
struct1.INPUT_JSON_KEYS = ["full story"]
struct1.OUTPUT_OBJECT_NAME = "leanings"
struct1.OUTPUT_JSON_KEYS = ["leaning"]
struct1.OUTPUT_JSON_KEY_DESCRIPTIONS = ["The leaning of the article (left, right, center, none)"]
struct1.OUTPUT_COLUMN_NAMES = ["gpt_leaning"]
struct1.MAX_ANTICIPATED_OUTPUT_WORDS = 10
struct1.MAX_ANTICIPATED_INPUT_WORDS = df['clean'].str.split().str.len().max()  # Set to the max length of the input column
struct1.jsonify()  

struct1.batch_size = machin.get_dynamic_batch_size(machin) #sets the max to fit in the context window



result_df, duration = Batcher.call_chunked(df, machin, struct1)
result_df.drop(columns=['clean'], inplace=True)
result_df = pd.concat([result_df, df_others])

result_df.to_csv('full_study_files/set2_sampled_withgpt4o.csv', index=False)