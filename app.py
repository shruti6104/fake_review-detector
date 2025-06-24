import streamlit as st
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model/vectorizer
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

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

# Streamlit UI
st.markdown("🕵️‍♀️ **Fake Review Detection System**")
st.markdown("Detect whether a product review is Genuine or Fake using NLP + Machine Learning.")

user_input = st.text_area("📝 Enter a product review:")

if st.button("🔍 Analyze Review"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]
        confidence = np.max(probability)

        if prediction == "CG" or confidence < 0.70:
            st.success("✅ Genuine Review Detected")
        else:
            st.error("❌ Fake Review Detected")

        st.info(f"🔎 Confidence: {confidence * 100:.2f}%")
