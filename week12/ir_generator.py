from sodigpt import sodiGPT
import os
import json
from time import sleep


interpretive_repertoires = ["Inspired", "Popular", "Moral", "Civic", "Economic", "Functional", "Ecological"]

# gpt = sodiGPT()
# response = gpt.call("hiiii")
# print(response)

all_irs = {}
for ir in interpretive_repertoires:
    plus_keywords = {
        "Inspired": ["righteous", "pre-ordained", "beautiful"],
        "Popular": ["preferred", "popular", "favorite"],
        "Moral": ["solidary", "responsible", "just"],
        "Civic": ["legal", "agreed", "common"],
        "Economic": ["beneficial", "economic", "affordable"],
        "Functional": ["effective", "necessary", "quick"],
        "Ecological": ["sustainable", "natural"]
    }[ir]
    minus_keywords = {
        "Inspired": ["false", "uncreative", "dull"],
        "Popular": ["resented", "feared", "isolated"],
        "Moral": ["inhumane", "asocial", "egoistic"],
        "Civic": ["scandalous", "unacceptable", "inappropriate"],
        "Economic": ["wasted", "costly", "unproductive"],
        "Functional": ["dysfunctional", "inefficient", "useless"],
        "Ecological": ["unnatural", "irreversible"]
    }[ir]

    prompt = f"""
        You are an expert in interpretive repertoires. Write two detailed first-person paragraphs (about 250 words each) describing the "likes" and "dislikes" of someone who exclusively uses the "{ir}" interpretive repertoire.

        - The first paragraph should focus on what this person likes, focused around the keywords: {plus_keywords}.
        - The second paragraph should focus on what this person dislikes, focused around the keywords: {minus_keywords}.

        Be genuine in your tone, as if the person is aware of their own interpretive lens. Use generic statements where possible that do not imply specific interests or contexts, but rather reflect a general worldview shaped by the "{ir}" interpretive repertoire.
        ```

    """

    # response = gpt.call(prompt)
    # resp = gpt.get_json_string_from_llmresp(response, delimiter="```")
    like_resp = input(prompt)
    minus_resp = input("Dislikes: ")
    if like_resp:
        try:
            all_irs[ir] = {
                "ir": ir,
                "likes_keywords": plus_keywords,
                "dislikes_keywords": minus_keywords,
                "likes": like_resp,
                "dislikes": minus_resp  
            }
        except Exception as e:
            print(f"Error parsing response for {ir}: {e}")
    else:
        print(f"No valid response received for {ir}")

    sleep(1)  # To avoid hitting rate limits



formatted_irs = []
for ir, data in all_irs.items():
    formatted_irs.append({
        "IR": data["ir"],
        "bucket": "+",
        "guidewords": data["likes_keywords"],
        "likes": data["likes"]
    })
    formatted_irs.append({
        "IR": data["ir"],
        "bucket": "-",
        "guidewords": data["dislikes_keywords"],
        "dislikes": data["dislikes"]
    })

# Save the results to a file
output_file = "interpretive_repertoires.json"
with open(output_file, 'w') as f:
    for item in formatted_irs:
        f.write(f"{item}\n")

