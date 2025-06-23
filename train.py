import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from settings import (
    TRAIN_FILE_PATH,
    HF_MODEL_NAME,
    SAVE_MODEL_PATH,
)


# 1. Load data
with open(TRAIN_FILE_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

data = [line.rsplit("|", 1) for line in lines]

df = pd.DataFrame(data, columns=["text", "label"])
df["label"] = df["label"].astype(int)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# 2. Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=2)


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
trainer.save_model(SAVE_MODEL_PATH)
tokenizer.save_pretrained(SAVE_MODEL_PATH)
