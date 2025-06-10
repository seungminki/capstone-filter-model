import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from preprocess import load_data, preprocess

model_name = "klue/bert-base"
train_file_path = "data/dataset.txt"
test_file_path = "validation.json"
save_model_path = "trained_model/"


# 1. Load data
with open(train_file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

data = [line.rsplit("|", 1) for line in lines]

df = pd.DataFrame(data, columns=["text", "label"])
df["label"] = df["label"].astype(int)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# 2. Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 3. Tokenize function
def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=128
    )


train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")


# 4. Metric
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds),
    }


# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# 7. Train!
trainer.train()

# 8. Save
trainer.save_model(save_model_path)
tokenizer.save_pretrained(save_model_path)


val_df = load_data(test_file_path)
val_df = preprocess(val_df)

new_texts = val_df["text"].to_list()
new_labels = val_df["is_active"].to_list()

model = model.to("cpu")

# 토크나이징
new_encodings = tokenizer(
    new_texts, padding=True, truncation=True, return_tensors="pt", max_length=128
)

# 모델을 평가 모드로 전환하고 예측
model.eval()
with torch.no_grad():
    outputs = model(**new_encodings)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy()

# print("📌 예측 결과:", preds)

if new_labels:
    acc = accuracy_score(new_labels, preds)
    f1 = f1_score(new_labels, preds)
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
