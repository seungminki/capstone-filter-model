import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import DataCollatorWithPadding
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# 파일 경로
file_path = "data/dataset.txt"

# 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# 각 줄을 "text|label" 형식으로 분리
data = [line.rsplit("|", 1) for line in lines]

# DataFrame 생성
df = pd.DataFrame(data, columns=["text", "label"])
df["label"] = df["label"].astype(int)


# 1. Load data
# df = df[:30000]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 2. Tokenizer & Model
model_name = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# 3. Tokenize function
def tokenize(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=128
    )


train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

# HuggingFace는 label 열을 int로 인식해야 함
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
)

# 7. Train!
trainer.train()

# 8. Save
trainer.save_model("abuse_model/")
tokenizer.save_pretrained("abuse_model/")
