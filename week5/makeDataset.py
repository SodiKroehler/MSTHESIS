import pandas as pd
import re
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import os
import numpy as np
from collections import Counter


def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    # text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def make_dataset():

    rdf = pd.read_csv("../raw/Qbias/allsides_balanced_news_headlines-texts.csv")

    subtitle = rdf.copy()
    subtitle = subtitle.dropna(subset=['heading'])
    subtitle['raw'] = '[HEADLINE]' + subtitle['heading'] + '[TEXT]' + subtitle['text']


    rdf = subtitle.copy()
    rdf = rdf[['raw', 'bias_rating']]
    rdf['clean'] = rdf['raw'].apply(clean_text)

    rdf['label_right'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'right' else 0)
    rdf['label_left'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'left' else 0)
    rdf['label_center'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'center' else 0)

    rdf.to_csv("week5_qbias_dataset.csv", index=False)


if __name__ == "__main__":
    make_dataset()