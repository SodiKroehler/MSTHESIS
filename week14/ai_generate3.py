from cardui import Machina, Structura, Batcher
import pandas as pd

machin = Machina("OpenAI", "gpt-4o")
machin.auth()
machin.max_retries = 1

#sources: 
source_leanings = {
    "nytimes.com": "lean_left",
    "msnbc.com": "left",
    "theguardian.com": "left",
    "reuters.com": "center",
    "apnews.com": "left",
    "npr.org": "lean_left",
    "foxnews.com": "right",
    "dailycaller.com": "right",
    "washingtonexaminer.com": "lean_right",
    "bbc.com": "center",
    "bbc.co.uk": "center",
    "aljazeera.com": "lean_left",
    "english.news.cn": "left",
}

df = pd.read_csv('set3_sampled_forAI.csv')
df['source_leaning'] = df['source'].map(source_leanings)
df['clean'] = 'HEADLINE: ' + df['human_clean_title'] + '\n\n' + 'STORY TEXT: ' + df['human_clean_text']
df.rename(columns={'gpt_justification': 'gpt_stance_justification', "gpt_bias_amnt": "gpt_poli_bias_amnt", "gpt_bias_cause": "gpt_poli_bias_cause"}, inplace=True)
df['gpt_poli_justification'] = None
# df = df.head(1)


struct1 = Structura()
struct1.PROMPT = "for each of the following headlines, determine if it is left-leaning (\"left\"), right-leaning (\"right\"), centrist with hardly any bias (\"center\") or not a polarized topic/could not be biased (\"none\"). Determine your answer, wait and think and see if it can be improved. Your answer for each should be only one word. #MY_INPUT_PLACEHOLDER"
struct1.INPUT_OBJECT_PLACEHOLDER = "#MY_INPUT_PLACEHOLDER"
struct1.INPUT_OBJECT_NAME = "stories"
struct1.INPUT_COLUMN_NAMES = ["clean"]
struct1.INPUT_JSON_KEYS = ["full story"]
struct1.OUTPUT_OBJECT_NAME = "ratings"
struct1.OUTPUT_JSON_KEYS = ["political_leaning","political_bias_amount", "political_bias_cause", "political_bias_justification", "article_subject", "ai_leaning", "ai_regulation_leaning", "immigration_leaning", "stance_justification"]
struct1.OUTPUT_JSON_KEY_DESCRIPTIONS = [
    "The leaning of the article ([FAR_LEFT, LEFT, LEAN_LEFT, CENTER, LEAN_RIGHT, RIGHT, FAR_RIGHT, UNDEFINED])",
    "The main subject or topic of the article ([IMMIGRATION, AI, AI_REGULATION, OTHER])",
    "bias_amnt",
    "bias_cause",
    "A justification or explanation for the assigned political bias ratings (around 250 words explaining the rationale for all the choices made)",
    "The article's leaning specifically on AI as a whole ([PRO_AI, ANTI_AI, NEUTRAL, UNDEFINED])",
    "The article's leaning specifically on AI regulation/ethics ([UNRESTRICTED, RESTRICTED, NEUTRAL, UNDEFINED])",
    "The article's leaning specifically on immigration issues ([PRO_IMM, ANTI_IMM, NEUTRAL, UNDEFINED])",
    "The amount or degree of bias present in the article ([LOW, MEDIUM, HIGH, UNDEFINED])",
    "The main cause or source of any bias detected ([WORD_CHOICE, CONTENT, FRAMING, OTHER, UNDEFINED])",
    "A justification or explanation for the assigned ratings (around 250 words explaining the rationale for all the stance choices made)"
]
struct1.OUTPUT_COLUMN_NAMES = ['gpt_poli_leaning', "gpt_poli_bias_amnt", "gpt_poli_bias_cause", "gpt_poli_justification", 'gpt_article_subject','gpt_ai_leaning', 'gpt_aireg_leaning', 'gpt_imm_leaning','gpt_stance_justification']

struct1.MAX_ANTICIPATED_INPUT_WORDS = df['clean'].str.split().str.len().max()  # Set to the max length of the input column
struct1.jsonify()  

struct1.batch_size = 1 #sets the max to fit in the context window

struct1.PROMPT = """
You are a media analyst evaluating political framing and bias in news headlines. For the news article below, complete the following tasks:

1. **Assess the political leaning of the article overall**:
   - Choose from: FAR_LEFT, LEFT, LEAN_LEFT, CENTER, LEAN_RIGHT, RIGHT, FAR_RIGHT, UNDEFINED
   - Select UNDEFINED only if the article is non-political or the bias is indiscernible.

2. **Evaluate the degree and nature of political bias**:
   - Bias Amount: NONE, LOW, MEDIUM, HIGH, UNDEFINED
   - Bias Cause: WORD_CHOICE | CONTENT | FRAMING | OTHER | UNDEFINED
     - Use WORD_CHOICE for overtly biased language.
     - Use CONTENT for selective inclusion or exclusion of facts.
     - Use FRAMING when the issue is shaped in a biased way without overt language.
     - Use OTHER if none of the above apply.
     - Use UNDEFINED only if this does not apply meaningfully.

3. **Identify the article’s primary subject**:
   - Choose from: IMMIGRATION | AI | AI_REGULATION | OTHER
   - If multiple topics are covered, select the most prominent.

4. **Classify the article’s stance (if relevant)**:
   - AI: PRO_AI | ANTI_AI | NEUTRAL | UNDEFINED
   - AI Regulation: PRO_REGULATION | ANTI_REGULATION | NEUTRAL | UNDEFINED
   - Immigration: PRO_IMMIGRATION | ANTI_IMMIGRATION | NEUTRAL | UNDEFINED
   - Use NEUTRAL if the article expresses both pro and anti views without a clear position.
   - Use UNDEFINED only if the topic is not addressed or the stance cannot be determined.

5. **Provide two short justifications (~250 words each)**:
   - One for your political bias evaluation (leaning, bias amount, and bias cause).
   - One for your stance classification (AI, AI Regulation, Immigration).

Before you respond, **pause and reevaluate your initial impression**. Is there a subtler interpretation worth considering?

Return your answer in valid JSON with this structure:

```json
{
  "ratings": [
    {
      "_ID": "the _ID of the input object",
      "political_leaning": "FAR_LEFT | LEFT | LEAN_LEFT | CENTER | LEAN_RIGHT | RIGHT | FAR_RIGHT | UNDEFINED",
      "political_bias_amount": "NONE | LOW | MEDIUM | HIGH | UNDEFINED",
      "political_bias_cause": "WORD_CHOICE | CONTENT | FRAMING | OTHER | UNDEFINED",
      "political_bias_justification": "Explanation of political leaning, bias level, and bias cause (~250 words)",
      "article_subject": "IMMIGRATION | AI | AI_REGULATION | OTHER",
      "ai_leaning": "PRO_AI | ANTI_AI | NEUTRAL | UNDEFINED",
      "ai_regulation_leaning": "PRO_REGULATION | ANTI_REGULATION | NEUTRAL | UNDEFINED",
      "immigration_leaning": "PRO_IMMIGRATION | ANTI_IMMIGRATION | NEUTRAL | UNDEFINED",
      "stance_justification": "Explanation of all stance ratings (~250 words)"
    }
  ]
}

The news article is: #MY_INPUT_PLACEHOLDER
```
"""


rdf, duration = Batcher.call_chunked(df, machin, struct1)
# answer = rdf.loc[0, 'LLM_RESPONSE']
# print(answer)
rdf.drop(columns=['clean'], inplace=True)
rdf.to_csv('set3_sampled_withgpt4o.csv', index=False)