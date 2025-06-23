import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from preprocess import load_data, preprocess
from settings import SAVE_MODEL_PATH, VALIDATION_FILE_PATH

val_df = load_data(VALIDATION_FILE_PATH)
val_df = preprocess(val_df)

new_texts = val_df["text"].to_list()
new_labels = val_df["is_active"].to_list()

tokenizer = AutoTokenizer.from_pretrained(SAVE_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    SAVE_MODEL_PATH, num_labels=2
)

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
