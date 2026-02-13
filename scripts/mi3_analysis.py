import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

df = pd.read_csv("letterboxd_top250_reviews_clean.csv")

df = df[df["review_text"].astype(str).str.strip().str.len() > 0].copy()

def stars_to_label(stars: float) -> str:
    if stars < 2.5:
        return "negative"
    elif stars <= 3.5:
        return "neutral"
    else:
        return "positive"

df["true_label"] = df["star_rating"].apply(stars_to_label)

model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipe = pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)

texts = df["review_text"].astype(str).tolist()
batch_size = 32
pred_labels = []
pred_scores = []

for i in tqdm(range(0, len(texts), batch_size), desc="Running HF sentiment"):
    batch = texts[i:i+batch_size]
    preds = sentiment_pipe(batch, truncation=True, max_length=512)
    pred_labels.extend([p["label"].lower() for p in preds])
    pred_scores.extend([p["score"] for p in preds])

df["pred_label"] = pred_labels
df["pred_score"] = pred_scores

def apply_neutral_threshold(label, score, low=0.4, high=0.6):
    if low <= score <= high:
        return "neutral"
    return label

df["pred_label_thresh"] = [
    apply_neutral_threshold(l, s, 0.4, 0.6) for l, s in zip(df["pred_label"], df["pred_score"])
]

y_true = df["true_label"]
y_pred = df["pred_label_thresh"]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 (macro):", f1_score(y_true, y_pred, average="macro"))
print("Confusion Matrix:", confusion_matrix(y_true, y_pred, labels=["negative","neutral","positive"]))
print(classification_report(y_true, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: HF Sentiment vs Star Ratings")
plt.show()

df["pred_label_thresh"].value_counts().plot(kind="bar")
plt.title("Distribution of HF Predicted Sentiment")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Count")
plt.show()

df.boxplot(column="star_rating", by="pred_label_thresh")
plt.title("Star Ratings by HF Predicted Sentiment")
plt.suptitle("")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Star Rating")
plt.show()

df["pred_score"].plot(kind="hist", bins=30)
plt.title("Distribution of HF Confidence Scores")
plt.xlabel("Confidence Score")
plt.ylabel("Count")
plt.show()

df["correct"] = df["true_label"] == df["pred_label_thresh"]

df.boxplot(column="pred_score", by="correct")
plt.title("Model Confidence: Correct vs Incorrect Predictions")
plt.suptitle("")
plt.xlabel("Prediction Correct?")
plt.ylabel("Confidence Score")
plt.show()

df["true_label"].value_counts().plot(kind="bar")
plt.title("True Sentiment Distribution (From Ratings)")
plt.show()

