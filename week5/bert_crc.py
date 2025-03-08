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
import time
from datetime import datetime


RANDOM_SEED = 42
TEST_SIZE = 0.2
SAMPLE_SIZE = 10

def train_anything(rdf, modelName, uncased=True, num_classes=2, label_col='label'):
    start_time = time.time()
    directory_path = "./"+modelName

    os.makedirs(directory_path, exist_ok=True)


    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf[label_col].tolist())
    

    train_df, test_df = train_test_split(rdf, test_size=0.2, random_state=42)

    CASEMENT_NAME = "distilbert-base-uncased" if uncased else "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(CASEMENT_NAME)
        

    def tokenize_data(examples):
        return tokenizer(examples["raw"] if not uncased else examples["clean"], truncation=True)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    tokenized_train = train_dataset.map(tokenize_data, batched=True)
    tokenized_test = test_dataset.map(tokenize_data, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(CASEMENT_NAME, num_labels=num_classes)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=directory_path+"/results",
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True
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

    #prediction

    predictions = trainer.predict(tokenized_test)

    logits = predictions.predictions 
    predicted_labels = np.argmax(logits, axis=1) 
    true_labels = test_df['label'].values  # True labels from dataset

    # Step 3: Compute Metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted")

    end_time = time.time()
    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [modelName, accuracy, precision, recall, f1, start_time, (start_time-end_time), end_date]

def run_all_combos():
    rdf = pd.read_csv("week5_qbias_dataset.csv")
    if not os.path.exists("week5_results.csv"):
        resPD = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Start_Time", "Duration", "End_Date"])
    else:
        resPD = pd.read_csv("week5_results.csv")
    results = []
    rdf.dropna(subset=['clean'], inplace=True)
    rdf = rdf[rdf['clean'].str.len() > 0]
    rdf = rdf.sample(SAMPLE_SIZE, random_state=RANDOM_SEED)
    
    
    for uncased in [True, False]:
        results.append(train_anything(rdf, f"multiclass_{'uncased' if uncased else 'cased'}", uncased, 3, 'bias_rating'))
    
    for uncased in [True, False]:
        results.append(train_anything(rdf, f"binary_left{'' if uncased else '_cased'}", uncased, 2, 'label_left'))

    #not left
    rdf['label_not_left'] = rdf['label_left'].apply(lambda x: 0 if x == 1 else 1)
    results.append(train_anything(rdf, f"binary_not_left", uncased, 2, 'label_not_left'))
    
    #right 
    results.append(train_anything(rdf, f"binary_right", uncased, 2, 'label_right'))

    #center
    results.append(train_anything(rdf, f"binary_center", uncased, 2, 'label_center'))

    newresPd = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Start_Time", "Duration", "End_Date"])
    pdResults = pd.concat([resPD, newresPd], ignore_index=True)
    pdResults.to_csv("week5_results.csv", index=False)



def eval_anything(rdf, modelName, label_encoder, labelColumn='label'):
    model_path = "./"+modelName+"/model"   

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    label_encoder = preprocessing.LabelEncoder()
    rdf['label'] = label_encoder.fit_transform(rdf['label_left'].tolist())
    rdf.dropna(subset=['clean'], inplace=True)

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

    true_labels = rdf['label'].values  # True labels

    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted")

    breakpoint()
    return [modelName, accuracy, precision, recall, f1]



    
if __name__ == "__main__":
    # run_all_combos()
    # eval_anything(pd.read_csv("week5_qbias_dataset.csv"), "binary_left", None, 'label_left')