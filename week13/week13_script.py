from cardui import Machina, Structura, Batcher, Utilitas
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

df = pd.read_csv('/Users/sodikroehler/Documents/WORKBENCH/MSTHESIS/MSTHESIS/raw/GDELT/june_revisit/combined_gdelt_filtered.csv')
df['source_leaning'] = df['source'].map(source_leanings)
df['clean'] = 'HEADLINE: ' + df['title'] + '\n\n' + 'STORY TEXT: ' + df['text']
# df = df.head(1)
df = df.groupby('source_leaning', group_keys=False).apply(lambda x: x.sample(n=100, random_state=42) if len(x) >= 100 else x).reset_index(drop=True)


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
struct1.jsonify()  

struct1.batch_size = machin.get_max_batch_size(struct1, df) #sets the max to fit in the context window



result_df, duration = Batcher.call_chunked(df, machin, struct1)
result_df.to_csv('leanings.csv', index=False)