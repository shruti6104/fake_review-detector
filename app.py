import streamlit as st
import joblib
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# === Download necessary NLTK resources (first run only) ===
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# === Preprocessing ===
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

# === Load model and vectorizer ===
model = joblib.load("fake_review_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# === Streamlit UI ===
st.set_page_config(page_title="🕵️‍♀️ Fake Review Detector", layout="centered")
st.title("🕵️‍♀️ Fake Review Detection System")
st.write("Detect whether a product review is **Genuine** or **Fake** using NLP + Machine Learning.")

st.markdown("---")

# === User Input ===
user_input = st.text_area("✍️ Enter a product review:", height=150)

if st.button("🔍 Analyze Review"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review to analyze.")
    else:
        try:
            cleaned = clean_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            proba = model.predict_proba(vector)[0]
            confidence = round(max(proba) * 100, 2)

            if prediction == 'CG':
                st.success("✅ Genuine Review Detected")
                st.info(f"🤖 Confidence: {confidence}%")
            else:
                st.error("❌ Fake Review Detected")
                st.info(f"🤖 Confidence: {confidence}%")

        except Exception as e:
            st.error("An error occurred while processing the review.")
            st.text(str(e))
