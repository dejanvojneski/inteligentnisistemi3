import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = 0
true["label"] = 1

df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
df["content"] = df["title"] + " " + df["text"]
df = df[["content", "label"]].dropna()

X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

acc = accuracy_score(y_test, model.predict(X_test_vec))
print(f"✅ Accuracy: {acc:.2%}")

os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
print("✅ Model saved!")