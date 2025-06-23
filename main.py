
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('TkAgg')

import os
import string
import nltk
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# === NLTK Downloads ===
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Step 1: Load Dataset ===
df = pd.read_csv("reviews.csv")
print("📋 Column names:", df.columns)

# === Step 2: Rename Columns if Needed ===
if 'text_' in df.columns:
    df.rename(columns={'text_': 'review'}, inplace=True)
if 'label' not in df.columns:
    raise ValueError("❌ 'label' column missing in dataset")

# === Step 3: Clean the Text ===
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned_review'] = df['review'].apply(clean_text)
print("\n🔍 Cleaned Reviews:\n")
print(df[['review', 'cleaned_review']].head())

# === Step 4: TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['label']

# === Step 5: Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 6: Model Training ===
model = LogisticRegression()
model.fit(X_train, y_train)

# === Step 7: Evaluation ===
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# === Step 8: Save Model and Vectorizer ===
joblib.dump(model, "fake_review_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved successfully.")

# === Step 9: Create Visualizations Folder ===
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# === Step 10: Word Cloud ===
all_words = ' '.join(df['cleaned_review'])
wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Most Common Words in Reviews")
plt.savefig("screenshots/wordcloud.png")
plt.show()

# === Step 11: Review Length Distribution ===
df['review_length'] = df['cleaned_review'].apply(lambda x: len(x.split()))
plt.figure(figsize=(8, 4))
plt.hist(df['review_length'], bins=30, color='skyblue', edgecolor='black')
plt.title("Review Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")
plt.savefig("screenshots/review_length.png")
plt.show()
