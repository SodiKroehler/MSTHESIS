import pandas as pd
# from langdetect import detect
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score
import seaborn as sns
import re
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader




class LlamaMultiClassClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(LlamaMultiClassClassifier, self).__init__()
        self.llama = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.llama.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token embedding
        logits = self.classifier(hidden_states)
        return logits




tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="auto")

class MultiClassDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_length=256):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        text = self.queries[idx]
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": self.labels[idx]
        }



rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
label_encoder = preprocessing.LabelEncoder()
rdf = rdf[rdf['repeatType'] == 'subtitle']

dataset = MultiClassDataset(rdf["clean"].tolist(), labels, tokenizer)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

label_encoder = LabelEncoder()
ldf["class_id"] = label_encoder.fit_transform(ldf["code"])

labels = torch.tensor(ldf["class_id"].values, dtype=torch.long)

label_mapping = {idx: label for idx, label in enumerate(label_encoder.classes_)}
