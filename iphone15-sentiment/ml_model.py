import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from clean import big_cleaning
from config import *

# Load data
data = pd.concat([
    pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Class%200'),
    pd.read_csv(f'https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Class%201')
], ignore_index=True)

# Preprocess data
x = data['text'].astype(str).apply(big_cleaning)
y = data['class'].astype(int)

# Binary Count Vectorization
binary_vectorizer = CountVectorizer(analyzer=lambda x: x, binary=True)
X_bin = binary_vectorizer.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X_bin, y, test_size=0.17, random_state=11)

classifier = RandomForestClassifier(n_estimators=600, random_state=11)
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print("Sentiment Analysis")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))
print(accuracy_score(y_test, y_pred))

# Test with Sentiment Phrases
test_phrases = pd.Series(["ผมว่าก็ไม่แย่นะครับ แต่โคตรแย่"]).astype(str).apply(big_cleaning)
test_phrases_vectorized = binary_vectorizer.transform(test_phrases)

sentiment_prediction = classifier.predict(test_phrases_vectorized)
predicted_sentiment = ["positive" if prob == 1 else "negative" for prob in sentiment_prediction]

print("Sentiment Prediction:", predicted_sentiment)