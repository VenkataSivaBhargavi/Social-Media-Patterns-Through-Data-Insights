import pandas as pd
import numpy as np
import re
import nltk
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')
stop_words = stopwords.words('english')


data = pd.read_csv("dataset.csv")



print("Dataset Loaded Successfully")
print(data.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)          # remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)       # remove special chars
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

# Apply cleaning
data['clean_text'] = data['Text'].apply(clean_text)


X = data['clean_text']
y = data['Sentiment']

vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully!")