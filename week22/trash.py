import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, PreTrainedModel
import torch.nn.functional as F

class MultiTaskBert(PreTrainedModel):
    """
    BERT backbone + multiple categorical embeddings + shared trunk + 2 heads:
      - 'multi'  : multi-class classification
      - 'binary' : binary classification (logit)
    """
    config_class = BertConfig

    def __init__(self, config, cat_cardinalities, cat_emb_dims, n_classes):
        super().__init__(config)

        # TEXT BACKBONE
        self.bert = BertModel(config)
        text_hidden_dim = config.hidden_size

        # CATEGORICAL EMBEDDINGS
        assert len(cat_cardinalities) == len(cat_emb_dims)
        self.cat_embs = nn.ModuleList([
            nn.Embedding(num_cats, emb_dim)
            for num_cats, emb_dim in zip(cat_cardinalities, cat_emb_dims)
        ])
        cat_total_dim = sum(cat_emb_dims)

        # SHARED TRUNK
        shared_dim = text_hidden_dim + cat_total_dim
        self.fc1 = nn.Linear(shared_dim, shared_dim)

        # HEADS
        self.heads = nn.ModuleDict({
            "multi": nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_dim // 2, n_classes)    # logits for classes
            ),
            "binary": nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Linear(shared_dim // 2, 1)           # single logit
            )
        })

        self.post_init()  # important for PreTrainedModel

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        cat_ids=None,          # [batch, num_cat_vars]
        labels_multi=None,     # [batch] long
        labels_binary=None,    # [batch] 0/1
        **kwargs
    ):
        # 1. TEXT ENCODING
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS representation (first token)
        h_text = bert_out.last_hidden_state[:, 0, :]     # [B, hidden]

        # 2. CATEGORICAL ENCODINGS
        # cat_ids: [B, num_cat_vars]
        cat_emb_list = []
        for i, emb_layer in enumerate(self.cat_embs):
            e_i = emb_layer(cat_ids[:, i])               # [B, emb_dim_i]
            cat_emb_list.append(e_i)
        e_cat = torch.cat(cat_emb_list, dim=-1)          # [B, sum(emb_dims)]

        # 3. SHARED REPRESENTATION
        h = torch.cat([h_text, e_cat], dim=-1)           # [B, shared_dim]
        h = torch.relu(self.fc1(h))                      # [B, shared_dim]

        # 4. HEADS
        logits_multi  = self.heads["multi"](h)           # [B, n_classes]
        logits_binary = self.heads["binary"](h).squeeze(-1)  # [B]

        loss = None
        if labels_multi is not None and labels_binary is not None:
            loss_fct_multi  = nn.CrossEntropyLoss()
            loss_fct_binary = nn.BCEWithLogitsLoss()
            loss_multi = loss_fct_multi(logits_multi, labels_multi)
            loss_binary = loss_fct_binary(
                logits_binary,
                labels_binary.float()
            )
            loss = loss_multi + loss_binary

        # Trainer expects at least 'loss' during training; you can also
        # pass extra outputs for eval/prediction.
        return {
            "loss": loss,
            "logits_multi": logits_multi,
            "logits_binary": logits_binary
        }
    
    
    
    
    
    from transformers import TrainingArguments, Trainer
import numpy as np

# ==== PREPARE TOKENIZER ====
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# ==== EXAMPLE: fill these with your data ====
# texts_train, texts_val      : list[str]
# cat_ids_train, cat_ids_val  : np.ndarray of shape [N, num_cat_vars] (int)
# y_multi_train, y_multi_val  : np.ndarray of ints (class indices)
# y_binary_train, y_binary_val: np.ndarray of 0/1

train_dataset = MultiTaskDataset(
    texts=texts_train,
    cat_ids=cat_ids_train,
    y_multi=y_multi_train,
    y_binary=y_binary_train,
    tokenizer=tokenizer,
    max_length=128,
)

eval_dataset = MultiTaskDataset(
    texts=texts_val,
    cat_ids=cat_ids_val,
    y_multi=y_multi_val,
    y_binary=y_binary_val,
    tokenizer=tokenizer,
    max_length=128,
)

# ==== MODEL INIT ====
cat_cardinalities = [85, 12, 3]              # example: topic, outlet, gender
cat_emb_dims      = [16, 8, 4]               # choose what you like
n_classes         = int(np.max(y_multi_train)) + 1

config = BertConfig.from_pretrained("bert-base-uncased")
model = MultiTaskBert(
    config=config,
    cat_cardinalities=cat_cardinalities,
    cat_emb_dims=cat_emb_dims,
    n_classes=n_classes,
)

# Optional: freeze BERT if you just want to train the head + cat embeddings
# for param in model.bert.parameters():
#     param.requires_grad = False

# ==== TRAINING ARGS ====
training_args = TrainingArguments(
    output_dir="./mt_bert_multicat",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# (Optional) metrics
def compute_metrics(eval_pred):
    # eval_pred.predictions is whatever we return from the model
    # Trainer will convert our dict outputs into a tuple;
    # by default, it will use the first element. Easiest hack:
    # run without metrics at first, or manually unpack if needed.
    return {}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=None,  # or compute_metrics
)

# ==== TRAIN ====
trainer.train()

from torch.utils.data import Dataset
from transformers import BertTokenizerFast

class MultiTaskDataset(Dataset):
    def __init__(self, texts, cat_ids, y_multi, y_binary, tokenizer, max_length=128):
        self.texts = texts
        self.cat_ids = cat_ids          # shape [N, num_cat_vars], tensor or np.array
        self.y_multi = y_multi
        self.y_binary = y_binary
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        cats = self.cat_ids[idx]        # e.g. np.array([...]) or tensor
        label_multi = self.y_multi[idx]
        label_binary = self.y_binary[idx]

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "cat_ids": torch.tensor(cats, dtype=torch.long),
            "labels_multi": torch.tensor(label_multi, dtype=torch.long),
            "labels_binary": torch.tensor(label_binary, dtype=torch.float),
        }
        return item


training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR),
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    seed=42,
    logging_steps=10,
    load_best_model_at_end = True
)

metric = evaluate.load("accuracy")

fold_metrics = []


def compute_metrics(eval_pred):
    logits, bias_labels, ambi_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    prec, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = metric.compute(predictions=predictions, references=labels)
    print(f"precision is {prec}, recall is {recall}, f1 is {f1}")
    fold_metrics.append({"prec": prec, "recall": recall, "f1": f1, "acc": acc})
    return acc

for fold, (train_idx, test_idx) in enumerate(kf.split(ldf_ds)):
    print(f"\n--- Fold {fold + 1}/{K} ---")
    
    train_data = ldf_ds.iloc[train_idx].reset_index(drop=True)
    test_data = ldf_ds.iloc[test_idx].reset_index(drop=True)

    train_dataset = Dataset.from_pandas(train_data).map(preprocess, batched=True)
    test_dataset = Dataset.from_pandas(test_data).map(preprocess, batched=True)
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=7)] # apparently stops the training if it doesn't improve after 3
        #do we want to increase???
    )

    
    trainer.train()

trainer.save_model(OUTPUT_DIR + "/model")
with open(OUTPUT_DIR+ "/fold_metrics.json", "w") as f:
    json.dump(fold_metrics, f, indent=4)
import evaluate

acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")


def compute_metrics(eval_pred):
    # eval_pred: transformers.EvalPrediction
    # eval_pred.predictions: usually logits (or tuple)
    # eval_pred.label_ids: ground-truth labels

    preds = eval_pred.predictions
    # If your model returns a tuple (e.g. (logits_multi, logits_binary)),
    # grab the first element (multi-class head)
    if isinstance(preds, tuple):
        logits = preds[0]
    else:
        logits = preds

    labels = eval_pred.label_ids
    # multi-class: take argmax over class dimension
    preds_class = logits.argmax(axis=-1)

    acc = acc_metric.compute(predictions=preds_class, references=labels)
    f1  = f1_metric.compute(predictions=preds_class, references=labels, average="macro")

    return {
        "accuracy": acc["accuracy"],
        "f1_macro": f1["f1"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,   # <— now enabled
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./mt_model_run",
    evaluation_strategy="epoch",        # eval every epoch
    save_strategy="epoch",              # save checkpoints every epoch
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",  # use our metric name
    greater_is_better=True,
    save_total_limit=1,                 # keep only best/latest checkpoint
    report_to=[],                       # no wandb/tensorboard spam on cluster
)


test_pred = trainer.predict(test_dataset)
# test_pred.predictions: logits (or tuple)
# test_pred.label_ids: test labels (multi-class)


import numpy as np
import json

preds = test_pred.predictions
if isinstance(preds, tuple):
    logits_multi = preds[0]   # first element = multi-class head
else:
    logits_multi = preds

labels = test_pred.label_ids
pred_classes = logits_multi.argmax(axis=-1)

test_acc = acc_metric.compute(predictions=pred_classes, references=labels)
test_f1  = f1_metric.compute(predictions=pred_classes, references=labels, average="macro")

results = {
    "test_accuracy": test_acc["accuracy"],
    "test_f1_macro": test_f1["f1"],
}
print(results)


import os
import shutil

final_dir = "./mt_model_final"

# Save model + tokenizer in HF format
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Zip it up
shutil.make_archive("mt_model_final", "zip", final_dir)
# This creates mt_model_final.zip


import glob

ckpt_pattern = os.path.join(training_args.output_dir, "checkpoint-*")
for ckpt_dir in glob.glob(ckpt_pattern):
    shutil.rmtree(ckpt_dir)

