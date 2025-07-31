import pandas as pd
import random
import re
from charm import CHARM

ir_labels = {
    0: "Inspired",
    1: "Popular",
    2: "Moral",
    3: "Civic",
    4: "Economic",
    5: "Functional",
    6: "Ecological"
}

data = {
    "url": ["https://www.washingtonexaminer.com/restoring-america/courage-strength-optimism/3419752/artificial-intelligence-us-trump-middle-east-trip/"],
    "pull": "july_all25_ai",
    "source": ["washingtonexaminer.com"],
    "title": ["Trump’s Middle East trip has helped strengthen AI deals in the region"],
    "content": ["""Jared Cohen, the young historian turned president of global affairs at Goldman Sachs, once said, “When people think of digital diplomacy, they think of government tweeting. It is not what it is. That is public diplomacy only.”

In the immediate aftermath of President Donald Trump’s triumphant swing through the Gulf states, the details of the administration’s artificial intelligence agreements remain under wraps. Yet, the broad contours are already clear. By pledging to build hyperscale data centers in Saudi Arabia and the United Arab Emirates stocked with cutting‑edge U.S. chips, Washington has publicly reaffirmed the global reach of the American AI stack — semiconductors, algorithms, and the data that feed them — at the expense of China’s rival offering, manifest most visibly in Huawei’s stalled contracts across the region.

The pageantry of this new entente has eclipsed a more technical but equally consequential decision: Trump’s revocation of the Biden‑era “AI Diffusion Network” export‑control regime. I criticized that framework when it appeared in January for the simple reason that it converted every overseas shipment of high‑end silicon into a Kafkaesque license chase. Understanding why the rule existed, however, is indispensable for grasping the reservations now voiced by some China hawks in Congress and even a few MAGA fellow‑travelers — and for designing something more rational in its place.

The story begins with the scaling laws that dominate contemporary machine learning: more computation reliably begets more capable models.
"""]
}
odf = pd.DataFrame(data)
df = pd.DataFrame()
#df needs to be rated based on sentence:
for idx, row in enumerate(odf):
    sentences = re.split(r'(?<=[.!?])\s+', row["content"])
    for sentence in sentences:
        row = {
            "url": row["url"],
            "pull": row["pull"],
            "source": row["source"],
            "title": row["title"],
            "text": sentence,
            "position": idx + 1,
        }
        df = df.append(row, ignore_index=True)

for ir_id, ir_name in ir_labels.items():
    df[f"coh_score_{ir_id}"] = random.uniform(0, 1)


# Example usage
if __name__ == "__main__":

    seed_topics = [
        ['republican', 'democrat', 'political'],
        ['ai'],
        ['regulation', 'ethics'],
        ['immigration', 'undocumented'],
    ]

    lda = CHARM(n_topics=3, seed_topics=seed_topics)
    topic_keywords, doc_assignments = lda.fit(df)

    print("\nTop words per topic:")
    for i, topic in enumerate(topic_keywords):
        print(f"Topic {i}: {topic}")

    print("\nDocument assignments:")
    for i, (text, topic) in enumerate(zip(df['text'], doc_assignments)):
        print(f"Doc {i}: Topic {topic} | {text}")

    

    #overall idea:
    1. 