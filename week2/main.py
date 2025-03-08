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
    # text = "The president is doing a great job"
    # level = 0.5
    # print(pol_bias_reward_func(text, level))
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

    fdf = pd.read_csv("./baly_classification_on_qbias2.csv")
    rdf = fdf[['title','bias_rating', 'baly_class', 'left', 'center', 'right']]