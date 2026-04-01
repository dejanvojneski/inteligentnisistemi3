import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# load dataset
df = pd.read_csv("../dataset/fake_news.csv")

# IMPORTANT: check column names
print(df.columns)

X = df["text"]
y = df["label"]

# vectorize
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2)

# train model
model = LogisticRegression()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")