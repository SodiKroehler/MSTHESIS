from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

bias_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

bias_model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")

tqdm.pandas()


def pol_bias_reward_func(text):
    inputs = bias_tokenizer(text, return_tensors="pt")
    labels = torch.tensor([0])
    outputs = bias_model(**inputs, labels=labels)
    loss, logits = outputs[:2]
    scores = logits.softmax(dim=-1)[0].tolist()
    return scores



def apply_baly(row):
    # return [-1, -1, -1]
    return pol_bias_reward_func(row['title'])


if __name__ == "__main__":
    # sample_texts = ["Title: The president is doing a great job! Heading: The president is doing a great job!", 
    #                 "Title: The president is doing a great job!",
    #                 "Title: The president is doing a terrible job! Heading: The president is doing a terrible job!",
    #                 "Title: The president is doing a terrible job!",
    #                 "Title: "]

    # title = "Congress Adjusts Funding for Social Programs Amid Budget Debate"
    # subtitle_right= "Runaway welfare spending drains taxpayers while politicians expand dependency, ignoring the burden on working Americans."
    # sub_left = "As billionaires hoard wealth, conservatives slash essential aid, pushing vulnerable families further into crisis."
    # sub_cent = "Lawmakers negotiate adjustments to federal social program funding, citing economic constraints and policy priorities."
    # sample_texts = [f"Title: {title}! Heading: {subtitle_right}",
    #                 f"Title: {title}! Heading: {sub_left}",
    #                 f"Title: {title}! Heading: {sub_cent}",
    #                 f"Title: {title}!",
    #                 f"Title: {title}! Heading: "]


    title ="Congress Debates New Border Security Measures Amid Immigration Surge"
    sub_left= "Fearmongering conservatives push cruel anti-immigrant policies, ignoring Americas proud history of welcoming the oppressed."
    sub_cent = "Lawmakers discuss strategies to manage border security and immigration challenges, weighing enforcement and humanitarian concerns."
    sub_right = "Bidens open-border disaster spirals out of control as illegal immigrants flood the country, straining resources and endangering citizens."
    sample_texts = [f"t: {title}! h: {sub_right}",
                f"t: {title}! h: {sub_left}",
                f"t: {title}! h: {sub_cent}",
                f"t: {title}!",
                f"t: {title}! h: "]
    
    for t in sample_texts:
        print(f"{t} ____ : ____ {pol_bias_reward_func(t)}")

    # rdf = pd.read_csv("../raw/Qbias/allsides_balanced_news_headlines-texts.csv")
    # sdf = rdf.sample(10)
    # sdf = rdf.copy()

    # sdf[['left', 'center', 'right']] =sdf.progress_apply(apply_baly, axis=1, result_type='expand')
    # sdf.to_csv("./baly_classification_on_qbias.csv", index=False)
    # print(sdf)

    # fdf = pd.read_csv("./baly_classification_on_qbias.csv")
    # fdf['baly_class'] = fdf[['left', 'center', 'right']].idxmax(axis=1)
    # fdf.to_csv("./baly_classification_on_qbias2.csv", index=False)
    
    # report = classification_report(fdf["bias_rating"], fdf["baly_class"], zero_division=0)
    # print(report)

    # fdf = pd.read_csv("./baly_classification_on_qbias2.csv")
    # rdf = fdf[['title','bias_rating', 'baly_class', 'left', 'center', 'right']]