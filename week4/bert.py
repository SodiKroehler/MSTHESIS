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


    headline = rdf.copy()
    headline = headline.dropna(subset=['title'])
    headline = headline.drop_duplicates(subset=['title'])
    headline['raw'] = '[HEADLINE]' + headline['title']
    headline['repeatType'] = 'headline'


    subtitle = rdf.copy()
    subtitle = subtitle.dropna(subset=['heading'])
    subtitle = subtitle.drop_duplicates(subset=['heading'])
    subtitle['raw'] = '[HEADLINE]' + subtitle['title'] + '[SUBTITLE]' + subtitle['heading']
    subtitle['repeatType'] = 'subtitle'

    texts = rdf.copy()
    texts = texts.dropna(subset=['text'])
    texts = texts.drop_duplicates(subset=['text'])
    texts['raw'] = '[HEADLINE]' + texts['title'] + '[SUBTITLE]' + subtitle['heading'] + '[TEXT]' + texts['text']
    texts['repeatType'] = 'text'

    rdf = pd.concat([headline, subtitle, texts])
    rdf = rdf[['raw', 'bias_rating', 'repeatType']]
    rdf = rdf.drop_duplicates(subset=['raw'])
    rdf['clean'] = rdf['raw'].apply(clean_text)

    rdf['label_right'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'right' else 0)
    rdf['label_left'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'left' else 0)
    rdf['label_center'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'center' else 0)

    rdf.to_csv("../raw/Qbias/week4_cleaned.csv", index=False)

def train_anything(rdf, modelName):
    
    directory_path = "./"+modelName

    os.makedirs(directory_path, exist_ok=True)        
    train_df, test_df = train_test_split(rdf, test_size=0.2, random_state=42)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_data(examples):
        return tokenizer(examples["clean"], truncation=True)

    tokenized_train = train_dataset.map(tokenize_data, batched=True)
    tokenized_test = test_dataset.map(tokenize_data, batched=True)


    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=directory_path+"/results",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(directory_path+"/model")



    predictions = trainer.predict(tokenized_test)

    logits = predictions.predictions 
    predicted_labels = np.argmax(logits, axis=1) 
    true_labels = test_df['label'].values  # True labels from dataset

    # Step 3: Compute Metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted")

    # Step 4: Print Results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def training_on_all_dataspits_multiclass():
    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf['bias_rating'].tolist())
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]
    
    # rdf = rdf[rdf['repeatType'] == 'subtitle']
    # rdf = rdf.sample(100) #start with small copy to be sure it works, comment out later
    train_anything(rdf, "model")

def eval_anything(rdf, modelName, label_encoder, labelColumn='label'):
    model_path = "./"+modelName+"/model"   

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Send model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    headlines = rdf["clean"].tolist()
    inputs = tokenizer(headlines, truncation=True, padding=True, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # Convert back to original bias categories
    decoded_labels = label_encoder.inverse_transform(predictions)

    
    true_labels = rdf[labelColumn].values  # True labels

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    

def eval_training_on_all_dataspits_multiclass():
    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf['bias_rating'].tolist())
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]
    
    rdf = rdf[rdf['repeatType'] == 'headline']
    # rdf = rdf.sample(100) #start with small copy to be sure it works, comment out later
    eval_anything(rdf, "allsplits_allLables", label_encoder)

def train_on_subtitle_left():
    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    label_encoder = preprocessing.LabelEncoder()
    rdf = rdf[rdf['repeatType'] == 'subtitle']
    # rdf = rdf[rdf['bias_rating'] == 'left']
    # rdf['label_left'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'left' else 0)
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]

    rdf['label'] = label_encoder.fit_transform(rdf['label_left'].tolist())
    
    
    # rdf = rdf.sample(100) #start with small copy to be sure it works, comment out later
    train_anything(rdf, "subtitle_left")



def train_on_subtitle_right():
    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    label_encoder = preprocessing.LabelEncoder()
    rdf = rdf[rdf['repeatType'] == 'subtitle']
    # rdf['label_right'] = rdf['bias_rating'].apply(lambda x: 1 if x == 'right' else 0)
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]

    rdf['label'] = label_encoder.fit_transform(rdf['label_right'].tolist())
    
    
    
    # rdf = rdf.sample(100) #start with small copy to be sure it works, comment out later
    train_anything(rdf, "subtitle_right")


def train_on_subtitle_center():
    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    label_encoder = preprocessing.LabelEncoder()
    rdf = rdf[rdf['repeatType'] == 'subtitle']

    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]

    rdf['label'] = label_encoder.fit_transform(rdf['label_center'].tolist())
    
    
    # rdf = rdf.sample(100) #start with small copy to be sure it works, comment out later
    train_anything(rdf, "subtitle_center")



def eval_on_subtitle_left():

    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf['bias_rating'].tolist())
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]
    
    rdf = rdf[rdf['repeatType'] == 'headline']
    eval_anything(rdf, 'subtitle_left', label_encoder, "label_left")

def eval_on_subtitle_right():

    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]

    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf['bias_rating'].tolist())
    
    
    rdf = rdf[rdf['repeatType'] == 'headline']
    eval_anything(rdf, 'subtitle_right', label_encoder, "label_right")

def eval_on_subtitle_center():

    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]

    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf['bias_rating'].tolist())
    
    
    rdf = rdf[rdf['repeatType'] == 'subtitle']
    eval_anything(rdf, 'subtitle_center', label_encoder, "label_center")


def coutnerStuff():
    
    rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf['bias_rating'].tolist())
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]
    
    rdf = rdf[rdf['repeatType'] == 'headline']

    
    model_path = "./subtitle_left/model"   

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Send model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    headlines = rdf["clean"].tolist()
    inputs = tokenizer(headlines, truncation=True, padding=True, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to predicted labels
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    pred_counts = Counter(predictions)

    # Convert back to original bias categories
    decoded_labels = label_encoder.inverse_transform(predictions)

    
    true_labels = rdf["label"].values  # True labels

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Count how many times each class is predicted


    print("Prediction Distribution:", pred_counts)
    print("Unique Predicted Labels:", np.unique(predictions))
    print("True Label Distribution:", Counter(true_labels))
      
    
if __name__ == "__main__":

    # make_dataset()
    #allSplits
    # eval_training_on_all_dataspits_multiclass()

    #subtitle_left
    # train_on_subtitle_left()
    # train_on_subtitle_right()
    # train_on_subtitle_center()
    # eval_on_subtitle_left()
    # eval_on_subtitle_right()
    eval_on_subtitle_center()
    # rdf = pd.read_csv("../raw/Qbias/week4_cleaned.csv")
    # label_encoder = preprocessing.LabelEncoder()
    # rdf['label'] = label_encoder.fit_transform(rdf['bias_rating'].tolist())
    # rdf.dropna(subset=['clean'], inplace=True)
    # rdf = rdf[rdf['clean'].str.len() > 0]
    
    # rdf = rdf[rdf['repeatType'] == 'subtitle']
    # # eval_anything(rdf, 'subtitle_left', label_encoder, "label_left")
    # modelName = 'subtitle_center'
    # labelColumn = 'label_center'
    # model_path = "./"+modelName+"/model"   

    # torch.cuda.empty_cache()

    # model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    # # Send model to device (GPU if available)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # headlines = rdf["clean"].tolist()
    # inputs = tokenizer(headlines, truncation=True, padding=True, return_tensors="pt").to(device)
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(**inputs)

    # # Convert logits to predicted labels
    # predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

    # # Convert back to original bias categories
    # decoded_labels = label_encoder.inverse_transform(predictions)

    
    # true_labels = rdf[labelColumn].values  # True labels

    # accuracy = accuracy_score(true_labels, predictions)
    # precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

    # print(f"Accuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
 