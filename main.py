
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Load Dataset ===
df = pd.read_csv("reviews.csv")
print("📋 Column names:", df.columns)

# Rename if needed
if 'text_' in df.columns:
    df.rename(columns={'text_': 'review'}, inplace=True)
if 'label' not in df.columns:
    raise ValueError("❌ 'label' column missing")

# === Clean the Text ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    keep = {"not", "very", "no"}
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words or word in keep]
    return ' '.join(words)

df['cleaned_review'] = df['review'].apply(clean_text)

# === Vectorize ===
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['label']

# === Train ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# === Save Model ===
joblib.dump(model, "fake_review_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved")

# === Visualizations ===
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

all_words = ' '.join(df['cleaned_review'])
wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Reviews")
plt.savefig("screenshots/wordcloud.png")
plt.show()

# Length distribution
df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 4))
plt.hist(df['review_length'], bins=30, color='skyblue', edgecolor='black')
plt.title("Review Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")
plt.savefig("screenshots/review_length.png")
plt.show()
